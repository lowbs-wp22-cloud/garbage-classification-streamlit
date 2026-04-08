import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
import os
import gdown
from tensorflow.keras.models import load_model

def load_bulky_model():
    if not os.path.exists("bulky_classifier.h5"):
        url = "https://drive.google.com/uc?id=1WmiUE5u7BmeTlSJYVREf6gd9U2S2l_Og"
        gdown.download(url, "bulky_classifier.h5", quiet=False)
    return load_model("bulky_classifier.h5")

bulky_model = load_bulky_model()
# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="AI Waste Classification", page_icon="♻️")

# =============================
# DATABASE
# =============================
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)
    
    # Admin table
    c.execute("""
    CREATE TABLE IF NOT EXISTS staff (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        staff_id TEXT UNIQUE,
        name TEXT,
        email TEXT,
        password TEXT
    )
    """)
    
    # Pickup Requests table
    c.execute("""
    CREATE TABLE IF NOT EXISTS pickup_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        predicted_label TEXT,
        confidence REAL,
        address TEXT,
        pickup_date TEXT,
        pickup_time_slot TEXT,
        note TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Redemptions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS redemptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        reward_name TEXT,
        points_used INTEGER,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_garbage_model():
    return tf.keras.models.load_model("FYP_general_waste.h5")

@st.cache_resource
def load_furniture_model():
    return tf.keras.models.load_model("bulky_classifier.h5")

# =============================
# SESSION STATE DEFAULTS
# =============================
for key in ["role", "user", "category", "reward_pending", "show_reward", "page"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================
# AUTH FUNCTIONS
# =============================
def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row and check_password_hash(row[0], password)

def signup_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    if c.fetchone():
        conn.close()
        return False
    hashed_password = generate_password_hash(password)
    c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
              (name, email, hashed_password))
    conn.commit()
    conn.close()
    return True
# =============================
# SIDEBAR NAVIGATION
# =============================
if st.session_state.user:
    with st.sidebar:
        st.title("Navigation")

        if st.session_state.role == "USER":
            page = st.radio(
                "Go to",
                ["Home", "Upload Waste", "Reward Status", "Pickup Scheduling", "Reward History","Redeem Rewards","Redemption History","Profile", "Logout"],
                key="user_page_nav"
            )
            st.session_state.page = page

        elif st.session_state.role == "ADMIN":
            page = st.radio(
                "Go to",
                ["Home", "Pending Rewards","Pickup Requests","Scheduled Pickups","Analytics", "Logout"],
                key="admin_page_nav"
            )
            st.session_state.page = page
# =============================
# ROLE SELECTION (FIXED)
# =============================
if st.session_state.role is None:
    st.subheader("Select Role")

    role_choice = st.radio(
        "Choose your role",
        ["USER", "ADMIN"],
        index=None   # 👈 IMPORTANT: no default selection
    )

    if role_choice:
        st.session_state.role = role_choice
        st.rerun()

# =============================
# ADMIN LOGIN / SIGNUP
# =============================
if st.session_state.role == "ADMIN" and st.session_state.user is None:
    st.subheader("ADMIN Login / Sign Up")
    option = st.radio("Choose an option", ["Login", "Sign Up"], key="admin_option")
    
    if option == "Login":
        staff_id = st.text_input("Staff ID", key="admin_login_id")
        password = st.text_input("Password", type="password", key="admin_login_pw")
        
        if st.button("Login"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT password FROM staff WHERE staff_id=?", (staff_id,))
            row = c.fetchone()
            conn.close()
            if row and check_password_hash(row[0], password):
                st.session_state.user = staff_id
                st.session_state.page = "Home"
                st.success("ADMIN login successful!")
                st.rerun()
            else:
                st.error("Invalid StaffID or Password")
                
    elif option == "Sign Up":
        staff_id = st.text_input("Staff ID", key="admin_signup_id")
        name = st.text_input("Name", key="admin_signup_name")
        email = st.text_input("Email", key="admin_signup_email")
        password = st.text_input("Password", type="password", key="admin_signup_pw")
        confirm = st.text_input("Confirm Password", type="password", key="admin_signup_confirm")
        
        if st.button("Sign Up"):
            if password != confirm:
                st.error("Passwords do not match")
            else:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT * FROM staff WHERE staff_id=?", (staff_id,))
                if c.fetchone():
                    st.error("StaffID already exists")
                else:
                    hashed = generate_password_hash(password)
                    c.execute("INSERT INTO staff (staff_id, name, email, password) VALUES (?,?,?,?)",
                              (staff_id, name, email, hashed))
                    conn.commit()
                    conn.close()
                    st.success("Admin Sign Up successful! Please login.")

# =============================
# USER LOGIN / SIGNUP
# =============================
elif st.session_state.role == "USER" and st.session_state.user is None:
    st.subheader("USER Login / Sign Up")
    option = st.radio("Choose an option", ["Login", "Sign Up"], key="user_option")
    
    if option == "Login":
        email = st.text_input("Email", key="user_login_email")
        password = st.text_input("Password", type="password", key="user_login_pw")
        
        if st.button("Login"):
            if login_user(email, password):
                st.session_state.user = email
                st.session_state.page = "Home"
                st.success("USER login successful!")
                st.rerun()
            else:
                st.error("Invalid Email or Password")
                
    elif option == "Sign Up":
        name = st.text_input("Name", key="user_signup_name")
        email = st.text_input("Email", key="user_signup_email")
        password = st.text_input("Password", type="password", key="user_signup_pw")
        confirm = st.text_input("Confirm Password", type="password", key="user_signup_confirm")
        
        if st.button("Sign Up"):
            if password != confirm:
                st.error("Passwords do not match")
            elif not name or not email or not password:
                st.error("All fields are required")
            elif signup_user(name, email, password):
                st.success("Sign Up successful! Please login.")
            else:
                st.error("Email already registered")
# =============================
# LOGOUT HANDLER
# =============================
if st.session_state.page == "Logout":
    for key in ["role", "user", "category", "reward_pending", "show_reward", "page"]:
        st.session_state[key] = None
    st.rerun()
# =============================
# ADMIN DASHBOARD
# =============================
elif st.session_state.role == "ADMIN" and st.session_state.user:

    if st.session_state.page == "Home":
        st.title("Admin Dashboard")
        st.write("Welcome, Admin.")
        st.write("Use the sidebar to manage the system.")

    elif st.session_state.page == "Pending Rewards":
        st.title("Admin Dashboard - Pending Rewards")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, user_email, points, status, station FROM rewards WHERE status='PENDING'")
        pending_rewards = c.fetchall()
        conn.close()

        if pending_rewards:
            for reward in pending_rewards:
                reward_id, user_email, points, status, station = reward
                st.write(f"**User:** {user_email} | **Points:** {points} | **Status:** {status} | **Station:** {station}")
                if st.button(f"APPROVE {reward_id}", key=f"approve_{reward_id}"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("UPDATE rewards SET status='APPROVED' WHERE id=?", (reward_id,))
                    conn.commit()
                    conn.close()
                    st.success(f"Reward for {user_email} approved!")
                    st.rerun()
        else:
            st.info("No pending rewards.")
            
    elif st.session_state.page == "Pickup Requests":
        st.title("Admin Dashboard - Pickup Requests")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, user_email, predicted_label, confidence, status, created_at
            FROM pickup_requests
            WHERE status = 'PENDING_APPROVAL'
            ORDER BY created_at DESC
        """)
        pickup_requests = c.fetchall()
        conn.close()

        if pickup_requests:
            for request in pickup_requests:
                request_id, user_email, predicted_label, confidence, status, created_at = request

                st.write(f"**Request ID:** {request_id}")
                st.write(f"**User:** {user_email}")
                st.write(f"**Predicted Item:** {predicted_label}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.write(f"**Status:** {status}")
                st.write(f"**Created At:** {created_at}")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(f"APPROVE PICKUP {request_id}", key=f"approve_pickup_{request_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            "UPDATE pickup_requests SET status='APPROVED' WHERE id=?",
                            (request_id,)
                        )
                        conn.commit()
                        conn.close()
                        st.success(f"Pickup request {request_id} approved!")
                        st.rerun()

                with col2:
                    if st.button(f"REJECT PICKUP {request_id}", key=f"reject_pickup_{request_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            "UPDATE pickup_requests SET status='REJECTED' WHERE id=?",
                            (request_id,)
                        )
                        conn.commit()
                        conn.close()
                        st.warning(f"Pickup request {request_id} rejected.")
                        st.rerun()

                st.markdown("---")
        else:
            st.info("No pending pickup requests.")
            
    elif st.session_state.page == "Scheduled Pickups":
        st.title("Admin Dashboard - Scheduled Pickups")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, user_email, predicted_label, address, pickup_date, pickup_time_slot, note, status
            FROM pickup_requests
            WHERE status = 'SCHEDULED'
            ORDER BY pickup_date ASC
        """)
        scheduled_pickups = c.fetchall()
        conn.close()

        if scheduled_pickups:
            for pickup in scheduled_pickups:
                request_id, user_email, predicted_label, address, pickup_date, pickup_time_slot, note, status = pickup

                st.write(f"**Request ID:** {request_id}")
                st.write(f"**User:** {user_email}")
                st.write(f"**Item:** {predicted_label}")
                st.write(f"**Address:** {address}")
                st.write(f"**Pickup Date:** {pickup_date}")
                st.write(f"**Time Slot:** {pickup_time_slot}")
                st.write(f"**Note:** {note if note else 'No additional note'}")
                st.write(f"**Status:** {status}")

                if st.button(f"MARK COMPLETED {request_id}", key=f"complete_pickup_{request_id}"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()

                    # Mark pickup request as completed
                    c.execute(
                        "UPDATE pickup_requests SET status='COMPLETED' WHERE id=?",
                        (request_id,)
                    )

                    # Add 30 points to rewards table after completion
                    c.execute(
                        "INSERT INTO rewards (user_email, points, status, station) VALUES (?, ?, ?, ?)",
                        (user_email, 30, "APPROVED", "Door-to-door pickup completed")
                    )

                    conn.commit()
                    conn.close()

                    st.success(f"Pickup request {request_id} marked as completed. 30 points added to user.")
                    st.rerun()

                st.markdown("---")
        else:
            st.info("No scheduled pickups.")
            
    elif st.session_state.page == "Analytics":
        st.title("Admin Dashboard - Analytics")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Total users
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]

        # Total rewards records
        c.execute("SELECT COUNT(*) FROM rewards")
        total_rewards = c.fetchone()[0]

        # Total points awarded
        c.execute("SELECT COALESCE(SUM(points), 0) FROM rewards")
        total_points_awarded = c.fetchone()[0]

        # Total pickup requests
        c.execute("SELECT COUNT(*) FROM pickup_requests")
        total_pickup_requests = c.fetchone()[0]

        # Pending pickup requests
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='PENDING_APPROVAL'")
        pending_pickups = c.fetchone()[0]

        # Approved pickup requests
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='APPROVED'")
        approved_pickups = c.fetchone()[0]

        # Scheduled pickups
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='SCHEDULED'")
        scheduled_pickups = c.fetchone()[0]

        # Completed pickups
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='COMPLETED'")
        completed_pickups = c.fetchone()[0]

        # Rejected pickups
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='REJECTED'")
        rejected_pickups = c.fetchone()[0]

        # Total redemptions
        c.execute("SELECT COUNT(*) FROM redemptions")
        total_redemptions = c.fetchone()[0]

        conn.close()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Users", total_users)
            st.metric("Total Rewards Records", total_rewards)
            st.metric("Total Points Awarded", total_points_awarded)
            st.metric("Total Pickup Requests", total_pickup_requests)
            st.metric("Pending Pickup Requests", pending_pickups)

        with col2:
            st.metric("Approved Pickup Requests", approved_pickups)
            st.metric("Scheduled Pickups", scheduled_pickups)
            st.metric("Completed Pickups", completed_pickups)
            st.metric("Rejected Pickup Requests", rejected_pickups)
            st.metric("Total Redemptions", total_redemptions)
# =============================
# USER FLOW
# =============================
elif st.session_state.role == "USER" and st.session_state.user:
    st.title("♻️ Smart Recycling Reward System")

    if st.session_state.page == "Home":
        st.subheader("Welcome")
        st.write("Welcome to the Smart Recycling Reward System.")
        st.write("Use the sidebar to navigate through the system.")

    elif st.session_state.page == "Upload Waste" and st.session_state.category is None:
        st.subheader("Select Category")
        category = st.radio("Choose waste type", ["General Waste", "Furniture"])
        if st.button("Continue"):
            st.session_state.category = category
            st.rerun()

    elif st.session_state.page == "Upload Waste" and st.session_state.reward_pending is None:
        st.subheader("Upload Image")

        if st.button("Change Waste Category"):
            st.session_state.category = None
            st.rerun()

        expected_furniture = None
        if st.session_state.category == "Furniture":
            st.info("Supported bulky categories: Bed, Chair, Fridge, Sofa, Table, TV, Wardrobe")
            expected_furniture = st.selectbox(
                "Select the bulky item type you are uploading",
                ["bed image", "chair image", "fridge image", "sofa image", "table image", "tv image", "wardrobe image"]
            )

        file = st.file_uploader("Upload garbage image", type=["jpg", "png", "jpeg"])

        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)

            if st.session_state.category == "General Waste":
                model = load_garbage_model()
                labels = ["Paper", "Plastic", "Metal", "Glass", "Cardboard", "Trash"]
            else:
                model = load_furniture_model()
                labels = [
                    "bed image",
                    "chair image",
                    "fridge image",
                    "sofa image",
                    "table image",
                    "tv image",
                    "wardrobe image"
                ]

            img = image.resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)

            try:
                pred = model.predict(arr, verbose=0)
                confidence = float(np.max(pred))
                index = np.argmax(pred)

                if index >= len(labels):
                    st.error("Prediction error: label mismatch")
                else:
                    result = labels[index]

                    st.success(f"Prediction Result: {result}")
                    st.info(f"Confidence: {confidence * 100:.2f}%")

                    # =============================
                    # GENERAL WASTE LOGIC
                    # =============================
                    if st.session_state.category == "General Waste":
                        if confidence >= 0.70:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute(
                                "INSERT INTO rewards (user_email, points, status, station) VALUES (?, ?, ?, ?)",
                                (st.session_state.user, 10, "PENDING", None)
                            )
                            conn.commit()
                            conn.close()

                            st.session_state.reward_pending = True
                            st.success("✅ 10 points added successfully.")

                            if st.button("Check Reward"):
                                st.session_state.show_reward = True
                                st.rerun()
                        else:
                            st.warning("⚠️ Low confidence. No reward given. Please try another image.")

                    # =============================
                    # BULKY WASTE LOGIC
                    # =============================
                    elif st.session_state.category == "Furniture":
                        if confidence < 0.85:
                            st.warning("⚠️ Low confidence for bulky item. No pickup request created. Please try another image.")
                        elif result != expected_furniture:
                            st.error("Unsupported bulky item or mismatched prediction. No pickup request created.")
                        else:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute(
                                """
                                INSERT INTO pickup_requests
                                (user_email, predicted_label, confidence, address, pickup_date, pickup_time_slot, note, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (st.session_state.user, result, confidence, None, None, None, None, "PENDING_APPROVAL")
                            )
                            conn.commit()
                            conn.close()

                            st.success("✅ Bulky waste request submitted successfully.")
                            st.info("Your bulky waste request is now pending admin approval.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif st.session_state.page == "Reward Status":
        st.subheader("🎁 Reward Status")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT points, status, station FROM rewards WHERE user_email=? ORDER BY id DESC LIMIT 1",
            (st.session_state.user,)
        )
        reward = c.fetchone()
        conn.close()

        status = None

        if reward:
            points, status, station = reward
            st.info(f"You earned **{points} points** (Status: {status})")
            if status == "PENDING":
                st.warning("Waiting for ADMIN approval...")
            elif status == "APPROVED":
                st.success(f"Reward Approved! Delivered to: {station}")
        else:
            st.info("No reward record found yet.")

        station = st.selectbox(
            "Choose nearby recycling station",
            ["EcoPoint Center", "GreenCycle Hub", "City Recycling Station"]
        )

        if reward and st.button("Confirm Delivery") and status == "APPROVED":
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "UPDATE rewards SET station=? WHERE user_email=? AND status='APPROVED'",
                (station, st.session_state.user)
            )
            conn.commit()
            conn.close()
            st.success("✅ Delivery confirmed!")
            st.session_state.reward_pending = None
            st.session_state.category = None
            st.session_state.show_reward = None
            st.rerun()
            
    elif st.session_state.page == "Pickup Scheduling":
        st.subheader("🚚 Pickup Scheduling")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, predicted_label, status
            FROM pickup_requests
            WHERE user_email=? AND status='APPROVED'
            ORDER BY created_at DESC
            LIMIT 1
        """, (st.session_state.user,))
        approved_request = c.fetchone()
        conn.close()

        if approved_request:
            request_id, predicted_label, status = approved_request

            st.info(f"Approved bulky item: {predicted_label}")
            st.success("Your pickup request has been approved. Please schedule your pickup.")

            address = st.text_area("Pickup Address")
            pickup_date = st.date_input("Select Pickup Date")
            pickup_time_slot = st.selectbox(
                "Select Pickup Time Slot",
                ["9:00 AM - 12:00 PM", "12:00 PM - 3:00 PM", "3:00 PM - 6:00 PM"]
            )
            note = st.text_area("Additional Note (Optional)")

            if st.button("Confirm Pickup Schedule"):
                if not address.strip():
                    st.error("Please enter your pickup address.")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        UPDATE pickup_requests
                        SET address=?, pickup_date=?, pickup_time_slot=?, note=?, status='SCHEDULED'
                        WHERE id=?
                    """, (address, str(pickup_date), pickup_time_slot, note, request_id))
                    conn.commit()
                    conn.close()

                    st.success("✅ Pickup scheduled successfully!")
                    st.rerun()
        else:
            st.info("No approved pickup request available for scheduling yet.")
            
    elif st.session_state.page == "Reward History":
        st.subheader("📜 Reward History")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, points, status, station
            FROM rewards
            WHERE user_email=?
            ORDER BY id DESC
        """, (st.session_state.user,))
        reward_history = c.fetchall()
        conn.close()

        if reward_history:
            total_points = sum(record[1] for record in reward_history)
            st.info(f"Total Points Collected: {total_points}")

            for reward in reward_history:
                reward_id, points, status, station = reward

                st.write(f"**Reward ID:** {reward_id}")
                st.write(f"**Points:** {points}")
                st.write(f"**Status:** {status}")
                st.write(f"**Source / Station:** {station if station else 'Not assigned yet'}")
                st.markdown("---")
        else:
            st.info("No reward history found yet.")
            
    elif st.session_state.page == "Redeem Rewards":
        st.subheader("🎁 Redeem Rewards")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Total earned points
        c.execute("SELECT COALESCE(SUM(points), 0) FROM rewards WHERE user_email=?", (st.session_state.user,))
        total_earned = c.fetchone()[0]

        # Total redeemed points
        c.execute("SELECT COALESCE(SUM(points_used), 0) FROM redemptions WHERE user_email=?", (st.session_state.user,))
        total_redeemed = c.fetchone()[0]

        conn.close()

        available_points = total_earned - total_redeemed

        st.info(f"Available Points: {available_points}")

        reward_catalog = [
            ("AEON Voucher RM10", 100),
            ("TNG Reload Pin RM8", 80),
            ("Shopee Voucher RM10", 100),
            ("GrabFood Voucher RM10", 100),
            ("Lazada Voucher RM10", 100)
        ]

        for reward_name, points_required in reward_catalog:
            st.write(f"### {reward_name}")
            st.write(f"Required Points: {points_required}")

            if available_points >= points_required:
                if st.button(f"Redeem {reward_name}", key=f"redeem_{reward_name}"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute(
                        """
                        INSERT INTO redemptions (user_email, reward_name, points_used, status)
                        VALUES (?, ?, ?, ?)
                        """,
                        (st.session_state.user, reward_name, points_required, "COMPLETED")
                    )
                    conn.commit()
                    conn.close()

                    st.success(f"✅ You have successfully redeemed {reward_name}.")
                    st.rerun()
            else:
                st.warning("Not enough points to redeem this reward.")

            st.markdown("---")

    elif st.session_state.page == "Redemption History":
        st.subheader("🧾 Redemption History")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT reward_name, points_used, status, created_at
            FROM redemptions
            WHERE user_email=?
            ORDER BY created_at DESC
        """, (st.session_state.user,))
        redemption_history = c.fetchall()
        conn.close()

        if redemption_history:
            total_spent = sum(record[1] for record in redemption_history)
            st.info(f"Total Points Spent: {total_spent}")

            for record in redemption_history:
                reward_name, points_used, status, created_at = record

                st.write(f"**Reward:** {reward_name}")
                st.write(f"**Points Used:** {points_used}")
                st.write(f"**Status:** {status}")
                st.write(f"**Redeemed At:** {created_at}")
                st.markdown("---")
        else:
            st.info("No redemption history found yet.")
            
    elif st.session_state.page == "Profile":
        st.subheader("👤 User Profile")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # User info
        c.execute("SELECT name, email FROM users WHERE email=?", (st.session_state.user,))
        user_info = c.fetchone()

        # Total earned points
        c.execute("SELECT COALESCE(SUM(points), 0) FROM rewards WHERE user_email=?", (st.session_state.user,))
        total_earned = c.fetchone()[0]

        # Total redeemed points
        c.execute("SELECT COALESCE(SUM(points_used), 0) FROM redemptions WHERE user_email=?", (st.session_state.user,))
        total_redeemed = c.fetchone()[0]

        # Total reward records
        c.execute("SELECT COUNT(*) FROM rewards WHERE user_email=?", (st.session_state.user,))
        total_reward_records = c.fetchone()[0]

        # Total pickup requests
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE user_email=?", (st.session_state.user,))
        total_pickup_requests = c.fetchone()[0]

        # Completed pickups
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE user_email=? AND status='COMPLETED'", (st.session_state.user,))
        completed_pickups = c.fetchone()[0]

        conn.close()

        available_points = total_earned - total_redeemed

        if user_info:
            name, email = user_info

            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Role:** USER")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Earned Points", total_earned)
                st.metric("Available Points", available_points)
                st.metric("Total Reward Records", total_reward_records)

            with col2:
                st.metric("Total Redeemed Points", total_redeemed)
                st.metric("Total Pickup Requests", total_pickup_requests)
                st.metric("Completed Pickups", completed_pickups)
        else:
            st.error("User profile not found.")

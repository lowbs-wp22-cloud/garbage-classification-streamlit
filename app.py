import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash

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
    
    # Rewards table
    c.execute("""
    CREATE TABLE IF NOT EXISTS rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        points INTEGER,
        status TEXT,
        station TEXT
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
    return tf.keras.models.load_model("garbage_classifier.h5")

@st.cache_resource
def load_furniture_model():
    return tf.keras.models.load_model("hcr_model.h5")

# =============================
# SESSION STATE DEFAULTS
# =============================
for key in ["role","user","category","reward_pending","show_reward"]:
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
# ADMIN DASHBOARD
# =============================
elif st.session_state.role == "ADMIN" and st.session_state.user:
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

# =============================
# USER FLOW
# =============================
elif st.session_state.role == "USER" and st.session_state.user:
    st.title("♻️ Smart Recycling Reward System")
    
    # CATEGORY SELECTION
    if st.session_state.category is None:
        st.subheader("Select Category")
        category = st.radio("Choose waste type", ["General Waste", "Furniture"])
        if st.button("Continue"):
            st.session_state.category = category
            st.rerun()
    
    # IMAGE UPLOAD & PREDICT
    elif st.session_state.reward_pending is None:
        st.subheader("Upload Image")
        file = st.file_uploader("Upload garbage image", type=["jpg","png","jpeg"])
        
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)
            
            if st.session_state.category == "General Waste":
                model = load_garbage_model()
                labels = ["Paper", "Plastic", "Metal", "Glass", "Organic", "Trash"]
            else:
                model = load_furniture_model()
                labels = ["Chair", "Table", "Sofa", "Bed", "Cabinet"]
            
            target_height, target_width = model.input_shape[1], model.input_shape[2]
            img = image.resize((target_width, target_height))
            arr = np.expand_dims(np.array(img)/255.0, axis=0)
            
            try:
                pred = model.predict(arr)
                result = labels[np.argmax(pred)]
                st.success(f"Prediction Result: {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            
            # CREATE PENDING REWARD
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO rewards VALUES (NULL, ?, ?, ?, ?)",
                      (st.session_state.user, 10, "PENDING", None))
            conn.commit()
            conn.close()
            st.session_state.reward_pending = True
            
            if st.button("Check Reward"):
                st.session_state.show_reward = True
                st.experimental_rerun()
    
    # REWARD PAGE
    elif st.session_state.reward_pending or st.session_state.get("show_reward"):
        st.subheader("🎁 Reward Status")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT points, status, station FROM rewards WHERE user_email=? ORDER BY id DESC LIMIT 1",
                  (st.session_state.user,))
        reward = c.fetchone()
        conn.close()
        
        if reward:
            points, status, station = reward
            st.info(f"You earned **{points} points** (Status: {status})")
            if status == "PENDING":
                st.warning("Waiting for ADMIN approval...")
            elif status == "APPROVED":
                st.success(f"Reward Approved! Delivered to: {station}")
        
        station = st.selectbox(
            "Choose nearby recycling station",
            ["EcoPoint Center", "GreenCycle Hub", "City Recycling Station"]
        )
        
        if st.button("Confirm Delivery") and status == "APPROVED":
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("UPDATE rewards SET station=? WHERE user_email=? AND status='APPROVED'",
                      (station, st.session_state.user))
            conn.commit()
            conn.close()
            st.success("✅ Delivery confirmed!")
            st.session_state.reward_pending = None
            st.session_state.category = None
            st.session_state.show_reward = None

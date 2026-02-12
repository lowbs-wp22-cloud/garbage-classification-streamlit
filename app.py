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
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)
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
# MODELS
# =============================
@st.cache_resource
def load_garbage_model():
    return tf.keras.models.load_model("garbage_classifier.h5")

@st.cache_resource
def load_furniture_model():
    return tf.keras.models.load_model("hcr_model.h5")

# =============================
# SESSION STATE
# =============================
for key in ["role", "user", "category", "reward_pending", "show_reward"]:
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

def signup_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    if c.fetchone():
        conn.close()
        return False
    hashed_password = generate_password_hash(password)
    c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
              (username, email, hashed_password))
    conn.commit()
    conn.close()
    return True

# =============================
# ROLE SELECTION
# =============================
if st.session_state.role is None:
    st.subheader("Select Role")
    st.session_state.role = st.radio("Choose your role", ["USER", "ADMIN"])

# =============================
# LOGIN / SIGNUP FOR BOTH ROLES
# =============================
if st.session_state.user is None:
    st.subheader(f"{st.session_state.role} Login or Sign Up")
    option = st.radio("Choose an option", ["Login", "Sign Up"])
    
    if option == "Login":
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_user(email, password):
                st.session_state.user = email
                st.success(f"Login successful as {st.session_state.role}!")
                st.rerun()
            else:
                st.error("Invalid login")

    elif option == "Sign Up":
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

        if st.button("Sign Up"):
            if password != confirm_password:
                st.error("Passwords do not match")
            elif not username or not email or not password:
                st.error("All fields are required")
            elif signup_user(username, email, password):
                st.success(f"Sign Up successful as {st.session_state.role}! Please login.")
            else:
                st.error("Email already registered")

# =============================
# ADMIN DASHBOARD AFTER LOGIN
# =============================
elif st.session_state.role == "ADMIN":
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
                st.experimental_rerun()
    else:
        st.info("No pending rewards.")


# =============================
# USER LOGIN / SIGNUP
# =============================
elif st.session_state.role == "USER" and st.session_state.user is None:
    st.subheader("Login or Sign Up")
    option = st.radio("Choose an option", ["Login", "Sign Up"])
    
    if option == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(email, password):
                st.session_state.user = email
                st.rerun()
            else:
                st.error("Invalid login")

    elif option == "Sign Up":
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if password != confirm_password:
                st.error("Passwords do not match")
            elif not username or not email or not password:
                st.error("All fields are required")
            elif signup_user(username, email, password):
                st.success("Sign Up successful! Please login.")
            else:
                st.error("Email already registered")

# =============================
# USER CATEGORY SELECTION
# =============================
elif st.session_state.category is None:
    st.subheader("Select Category")
    category = st.radio("Choose waste type", ["General Waste", "Furniture"])
    if st.button("Continue"):
        st.session_state.category = category
        st.rerun()

# =============================
# USER IMAGE UPLOAD & PREDICT
# =============================
elif st.session_state.reward_pending is None:
    st.subheader("Upload Image")

    file = st.file_uploader("Upload garbage image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.session_state.category == "General Waste":
            model = load_garbage_model()
            labels = ["Paper", "Plastic", "Metal", "Glass", "Organic", "Trash"]
        else:
            model = load_furniture_model()
            labels = ["Chair", "Table", "Sofa", "Bed", "Cabinet"]

        # Dynamic model input size
        target_height, target_width = model.input_shape[1], model.input_shape[2]
        img = image.resize((target_width, target_height))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        try:
            pred = model.predict(arr)
            result = labels[np.argmax(pred)]
            st.success(f"Prediction Result: {result}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        # Create pending reward
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO rewards VALUES (NULL, ?, ?, ?, ?)",
            (st.session_state.user, 10, "PENDING", None)
        )
        conn.commit()
        conn.close()
        st.session_state.reward_pending = True

        # Button to manually check reward
        if st.button("Check Reward"):
            st.session_state.show_reward = True
            st.rerun()

# =============================
# USER REWARD PAGE
# =============================
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
        else:
            st.success(f"Reward Approved! Delivered to: {station}")
    else:
        st.info("No rewards yet.")

    station = st.selectbox(
        "Choose nearby recycling station",
        ["EcoPoint Center", "GreenCycle Hub", "City Recycling Station"]
    )

    if st.button("Confirm Delivery") and status == "APPROVED":
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE rewards
            SET station=?
            WHERE user_email=? AND status='APPROVED'
        """, (station, st.session_state.user))
        conn.commit()
        conn.close()
        st.success("✅ Delivery confirmed!")
        st.session_state.reward_pending = None
        st.session_state.category = None
        st.session_state.show_reward = None

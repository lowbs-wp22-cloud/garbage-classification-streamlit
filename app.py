import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
import os

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
    return tf.keras.models.load_model("FYP_general_waste.h5")

@st.cache_resource
def load_furniture_model():
    return tf.keras.models.load_model("hcr_model.h5")

# =============================
# SESSION STATE
# =============================
for key in ["user", "category", "reward_pending"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================
# AUTH
# =============================
def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row and check_password_hash(row[0], password)

# =============================
# UI HEADER
# =============================
st.title("♻️ Smart Recycling Reward System")

# =============================
# ROLE SELECTION
# =============================
if "role" not in st.session_state:
    st.subheader("Select Role")
    st.session_state.role = st.radio("Choose your role", ["USER", "ADMIN"])

# =============================
# ADMIN VIEW
# =============================
if st.session_state.role == "ADMIN":
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
# LOGIN / SIGNUP
# =============================
if st.session_state.user is None:
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
            else:
                # Check if email already exists
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE email=?", (email,))
                if c.fetchone():
                    st.error("Email already registered")
                else:
                    hashed_password = generate_password_hash(password)
                    c.execute(
                        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, hashed_password)
                    )
                    conn.commit()
                    conn.close()
                    st.success("Sign Up successful! Please login.")

# =============================
# CATEGORY
# =============================
elif st.session_state.category is None:
    st.subheader("Select Category")
    category = st.radio("Choose waste type", ["General Waste", "Furniture"])

    if st.button("Continue"):
        st.session_state.category = category
        st.rerun()

# =============================
# UPLOAD & PREDICT
# =============================
elif st.session_state.reward_pending is None:
    st.subheader("Upload Image")

    file = st.file_uploader("Upload garbage image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

        img = image.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        if st.session_state.category == "General Waste":
            model = load_garbage_model()
            labels = ["Paper", "Plastic", "Metal", "Glass", "Organic", "Trash"]
        else:
            model = load_furniture_model()
            labels = ["Chair", "Table", "Sofa", "Bed", "Cabinet"]

        pred = model.predict(arr)
        result = labels[np.argmax(pred)]

        st.success(f"Prediction Result: {result}")

        # ----- CREATE PENDING REWARD (but don't switch page automatically) -----
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO rewards VALUES (NULL, ?, ?, ?, ?)",
            (st.session_state.user, 10, "PENDING", None)
        )
        conn.commit()
        conn.close()

        st.session_state.reward_pending = True

        # ----- Add a button to check reward manually -----
        if st.button("Check Reward"):
            st.session_state.show_reward = True
            st.rerun()
# =============================
# REWARD PAGE
# =============================
elif st.session_state.reward_pending or st.session_state.get("show_reward"):
    st.subheader("🎁 Reward Status")

    st.info("You earned **10 points** (Status: PENDING)")

    station = st.selectbox(
        "Choose nearby recycling station",
        ["EcoPoint Center", "GreenCycle Hub", "City Recycling Station"]
    )

    if st.button("Confirm Delivery"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE rewards
            SET status='EARNED', station=?
            WHERE user_email=? AND status='PENDING'
        """, (station, st.session_state.user))
        conn.commit()
        conn.close()

        st.success("✅ Points earned successfully!")
        st.session_state.reward_pending = None
        st.session_state.category = None
        st.session_state.show_reward = None

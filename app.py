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
st.set_page_config(
    page_title="AI Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# =============================
# DATABASE SETUP
# =============================
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =============================
# LOAD MODELS (ONCE)
# =============================
GARBAGE_MODEL_PATH = "garbage_classifier.h5"
FURNITURE_MODEL_PATH = "hcr_model.h5"

@st.cache_resource
def load_garbage_model():
    return tf.keras.models.load_model(GARBAGE_MODEL_PATH)

@st.cache_resource
def load_furniture_model():
    return tf.keras.models.load_model(FURNITURE_MODEL_PATH)

# =============================
# SESSION STATE
# =============================
if "user" not in st.session_state:
    st.session_state.user = None

if "category" not in st.session_state:
    st.session_state.category = None

# =============================
# AUTH FUNCTIONS
# =============================
def signup_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, generate_password_hash(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    return result and check_password_hash(result[0], password)

# =============================
# UI HEADER
# =============================
st.title("♻️ AI Waste Classification System")
st.caption("Final Year Project – Multi-Model Streamlit Application")

# =============================
# LOGIN / SIGNUP
# =============================
if st.session_state.user is None:
    menu = st.sidebar.radio("Navigation", ["Login", "Sign Up"])

    if menu == "Login":
        st.subheader("🔐 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(email, password):
                st.session_state.user = email
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid email or password")

    else:
        st.subheader("📝 Create Account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            if signup_user(username, email, password):
                st.success("Account created successfully. Please login.")
            else:
                st.error("Email already exists")

# =============================
# CATEGORY SELECTION
# =============================
elif st.session_state.category is None:
    st.success(f"Logged in as: {st.session_state.user}")
    st.subheader("📂 Select Waste Category")

    category = st.radio(
        "Choose the category before uploading image:",
        ("General Waste", "Furniture")
    )

    if st.button("Continue"):
        st.session_state.category = category
        st.rerun()

# =============================
# UPLOAD & AUTO-PREDICT
# =============================
else:
    st.sidebar.success(f"User: {st.session_state.user}")
    st.sidebar.info(f"Category: {st.session_state.category}")

    if st.sidebar.button("Change Category"):
        st.session_state.category = None
        st.rerun()

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.category = None
        st.rerun()

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # -------- MODEL SELECTION --------
            if st.session_state.category == "General Waste":
                model = load_garbage_model()
                labels = ["Paper", "Plastic", "Metal", "Glass", "Organic", "Trash"]
            else:
                model = load_furniture_model()
                labels = ["Chair", "Table", "Sofa", "Bed", "Cabinet"]

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            predicted_label = labels[class_index]

        st.success(f"🧠 Prediction Result: {predicted_label}")

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
    page_title="AI Garbage Classification",
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
# LOAD MODEL (ONCE)
# =============================
MODEL_PATH = "/content/drive/MyDrive/FYP/garbage_classifier.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =============================
# SESSION STATE
# =============================
if "user" not in st.session_state:
    st.session_state.user = None

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
    if result and check_password_hash(result[0], password):
        return True
    return False

# =============================
# UI
# =============================
st.title("♻️ AI Garbage Classification System")
st.caption("Final Year Project – Streamlit Interface")

if st.session_state.user is None:
    menu = st.sidebar.radio("Navigation", ["Login", "Sign Up"])

    # ---------- LOGIN ----------
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

    # ---------- SIGN UP ----------
    else:
        st.subheader("📝 Create Account")

        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            if not username or not email or not password:
                st.warning("Please fill in all fields")
            elif signup_user(username, email, password):
                st.success("Account created successfully. Please login.")
            else:
                st.error("Email already exists")

# =============================
# UPLOAD & PREDICT
# =============================
else:
    st.sidebar.success(f"Logged in as {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    st.subheader("📤 Upload Garbage Image")

    uploaded_file = st.file_uploader(
        "Choose an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Running prediction..."):
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)

                labels = ["Paper", "Plastic", "Metal", "Glass", "Organic", "Trash"]
                predicted_label = labels[class_index]

                category = (
                    "Recyclable" if predicted_label != "Trash"
                    else "Non-Recyclable"
                )

            st.success(f"Category: {category}")
            st.info(f"Details: {predicted_label}")

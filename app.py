import streamlit as st
import sqlite3
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.security import generate_password_hash, check_password_hash

# =====================================================
# CONFIG
# =====================================================
DB_PATH = "garbage_app.db"
st.set_page_config(page_title="Garbage Classification System", layout="wide")

# =====================================================
# LOAD MODEL (ONLY ONE MODEL)
# =====================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("general_waste.h5")

model = load_model()

# =====================================================
# DATABASE INIT
# =====================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        category TEXT,
        station TEXT,
        points INTEGER,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(image):
    image = image.resize((224, 224))  # Change ONLY if your model uses different size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================================================
# PREDICTION FUNCTION (STRICTLY FOLLOW MODEL OUTPUT)
# =====================================================
def predict_garbage(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_index, confidence

# =====================================================
# SESSION DEFAULTS
# =====================================================
defaults = {
    "user": None,
    "user_role": None,
    "login_type": "User",
    "nav": "User Profile"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# AUTH FUNCTIONS
# =====================================================
def signup_user(username, email, password, role):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
            (username, email, generate_password_hash(password), role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password, role):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT password FROM users WHERE email=? AND role=?",
        (email, role)
    )
    row = c.fetchone()
    conn.close()
    return row and check_password_hash(row[0], password)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("♻️ Garbage Classification")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user_role.upper()}")

    st.session_state.nav = st.sidebar.radio(
        "Navigation",
        ["User Profile", "Upload Garbage"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# =====================================================
# LOGIN PAGE
# =====================================================
if not st.session_state.user:
    st.title("🔐 Login / Signup")

    st.radio("Login as:", ["User", "Admin"], horizontal=True, key="login_type")
    role = st.session_state.login_type.lower()

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(email, password, role):
                st.session_state.user = email
                st.session_state.user_role = role
                st.rerun()
            else:
                st.error("Invalid credentials")

    with signup_tab:
        username = st.text_input("Username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            if signup_user(username, email, password, role):
                st.session_state.user = email
                st.session_state.user_role = role
                st.rerun()
            else:
                st.error("Email already exists")

# =====================================================
# USER DASHBOARD
# =====================================================
elif st.session_state.user_role == "user":

    if st.session_state.nav == "User Profile":
        st.title("👤 User Profile")
        st.write(f"Email: {st.session_state.user}")
        st.write("Role: User")

    elif st.session_state.nav == "Upload Garbage":
        st.title("📷 Upload Garbage")

        category = st.radio(
            "Choose category:",
            ["General Waste", "Furniture"],
            horizontal=True
        )

        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

            if st.button("🔍 Predict"):
                with st.spinner("Analyzing image..."):
                    class_index, confidence = predict_garbage(image)

                st.success(f"Predicted Class Index: {class_index}")
                st.info(f"Confidence: {confidence*100:.2f}%")

# =====================================================
# ADMIN DASHBOARD
# =====================================================
elif st.session_state.user_role == "admin":
    st.title("🛠 Admin Dashboard")
    st.info("Admin features here")

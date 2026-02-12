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
# LOAD MODEL (ONLY general_waste.h5)
# =====================================================
@st.cache_resource
def load_general_model():
    return tf.keras.models.load_model("FYP_general_waste.h5")

general_model = load_general_model()

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

    conn.commit()
    conn.close()

init_db()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(image):
    image = image.resize((224, 224))   # Change if your model uses different size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================================================
# PREDICTION FUNCTION
# (No class name mapping, follow model output directly)
# =====================================================
def predict_general_waste(image):
    processed = preprocess_image(image)
    prediction = general_model.predict(processed)

    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return predicted_class, confidence

# =====================================================
# SESSION DEFAULTS
# =====================================================
defaults = {
    "user": None,
    "user_role": None,
    "login_type": "User",
    "page": "category",
    "selected_category": None
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
                st.session_state.page = "category"
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
                st.session_state.page = "category"
                st.rerun()
            else:
                st.error("Email already exists")

# =====================================================
# USER FLOW
# =====================================================
elif st.session_state.user_role == "user":

    # ---------------- CATEGORY PAGE ----------------
    if st.session_state.page == "category":

        st.title("📂 Choose Garbage Category")

        col1, col2 = st.columns(2)

        if col1.button("🗑 General Waste"):
            st.session_state.selected_category = "General Waste"
            st.session_state.page = "upload"
            st.rerun()

        if col2.button("🪑 Furniture"):
            st.session_state.selected_category = "Furniture"
            st.session_state.page = "upload"
            st.rerun()

    # ---------------- UPLOAD PAGE ----------------
    elif st.session_state.page == "upload":

        st.title(f"📷 Upload Image - {st.session_state.selected_category}")

        if st.button("⬅ Back to Category"):
            st.session_state.page = "category"
            st.rerun()

        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

            if st.button("🔍 Predict"):

                if st.session_state.selected_category == "General Waste":

                    with st.spinner("Analyzing image..."):
                        predicted_class, confidence = predict_general_waste(image)

                    st.markdown(f"### 🧠 Predicted Garbage : {predicted_class}")
                    st.markdown(f"### 📊 Confidence : {confidence*100:.2f}%")

                else:
                    st.warning("Furniture model not implemented yet.")

# =====================================================
# ADMIN DASHBOARD
# =====================================================
elif st.session_state.user_role == "admin":
    st.title("🛠 Admin Dashboard")
    st.info("Admin features can be implemented here.")

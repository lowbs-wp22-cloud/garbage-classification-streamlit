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

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        category TEXT,
        result TEXT
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
    return tf.keras.models.load_model("general_waste.h5")

@st.cache_resource
def load_furniture_model():
    return tf.keras.models.load_model("hcr_model.h5")

# =============================
# SESSION STATE
# =============================
for key, default in {
    "user": None,
    "auth_mode": "Login",
    "category": None,
    "current_page": "upload",
    "reward_created": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =============================
# AUTH HELPERS
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
    c.execute("SELECT password FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row and check_password_hash(row[0], password)

# =============================
# SIDEBAR (AFTER LOGIN)
# =============================
if st.session_state.user:
    st.sidebar.title("👤 User Dashboard")
    st.sidebar.write(f"**Email:** {st.session_state.user}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT COALESCE(SUM(points), 0)
        FROM rewards
        WHERE user_email=? AND status='EARNED'
    """, (st.session_state.user,))
    total_points = c.fetchone()[0]

    st.sidebar.metric("⭐ Total Points", total_points)

    st.sidebar.subheader("📜 Upload History")
    c.execute("""
        SELECT category, result
        FROM history
        WHERE user_email=?
        ORDER BY id DESC
        LIMIT 5
    """, (st.session_state.user,))
    history = c.fetchall()
    conn.close()

    if history:
        for cat, res in history:
            st.sidebar.write(f"- **{cat}** → {res}")
    else:
        st.sidebar.caption("No history yet")

    if st.sidebar.button("🚪 Logout"):
        for k in ["user", "category", "current_page", "reward_created"]:
            st.session_state[k] = None if k == "user" else "upload"
        st.session_state.auth_mode = "Login"
        st.rerun()

# =============================
# UI HEADER
# =============================
st.title("♻️ Smart Recycling Reward System")

# =============================
# LOGIN / SIGN UP CHOICE
# =============================
if st.session_state.user is None:
    st.subheader("Welcome")

    st.radio(
        "Choose an option",
        ["Login", "Sign Up"],
        key="auth_mode",
        horizontal=True
    )

    # ---------- LOGIN ----------
    if st.session_state.auth_mode == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(email, password):
                st.session_state.user = email
                st.rerun()
            else:
                st.error("Invalid email or password")

    # ---------- SIGN UP ----------
    else:
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            if not username or not email or not password:
                st.warning("Please fill in all fields")
            elif signup_user(username, email, password):
                st.success("Account created successfully. Please login.")
                st.session_state.auth_mode = "Login"
            else:
                st.error("Email already exists")

# =============================
# CATEGORY SELECTION
# =============================
elif st.session_state.category is None:
    st.subheader("Select Category")
    category = st.radio("Choose waste type", ["General Waste", "Furniture"])

    if st.button("Continue"):
        st.session_state.category = category
        st.session_state.current_page = "upload"
        st.rerun()

# =============================
# UPLOAD & PREDICT
# =============================
elif st.session_state.current_page == "upload":
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
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

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO history VALUES (NULL, ?, ?, ?)",
            (st.session_state.user, st.session_state.category, result)
        )

        if not st.session_state.reward_created:
            c.execute(
                "INSERT INTO rewards VALUES (NULL, ?, ?, ?, ?)",
                (st.session_state.user, 10, "PENDING", None)
            )
            st.session_state.reward_created = True

        conn.commit()
        conn.close()

        if st.button("🎁 Check Reward"):
            st.session_state.current_page = "reward"
            st.rerun()

# =============================
# REWARD PAGE
# =============================
elif st.session_state.current_page == "reward":
    st.subheader("🎁 Reward Page")

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

        st.session_state.category = None
        st.session_state.current_page = "upload"
        st.session_state.reward_created = False
        st.rerun()

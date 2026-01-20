import streamlit as st
import sqlite3
import random
from werkzeug.security import generate_password_hash, check_password_hash

# =====================================================
# CONFIG
# =====================================================
DB_PATH = "garbage_app.db"
st.set_page_config(page_title="Garbage Classification System", layout="wide")

# =====================================================
# DATABASE INIT + MIGRATION
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

    c.execute("PRAGMA table_info(rewards)")
    columns = [col[1] for col in c.fetchall()]

    if "category" not in columns:
        c.execute("ALTER TABLE rewards ADD COLUMN category TEXT")
    if "station" not in columns:
        c.execute("ALTER TABLE rewards ADD COLUMN station TEXT")

    conn.commit()
    conn.close()

init_db()

# =====================================================
# SESSION STATE INIT
# =====================================================
defaults = {
    "user": None,
    "user_role": None,
    "login_type": "User",
    "page": "upload",        # upload | station
    "nav": "User Profile",   # sidebar navigation
    "category": None,
    "prediction": None,
    "confidence": None
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
# SIMULATED AI PREDICTION
# =====================================================
def predict_garbage(category):
    if category == "General Waste":
        classes = ["Plastic", "Paper", "Metal", "Organic Waste"]
    else:
        classes = ["Chair", "Table", "Sofa", "Cabinet"]

    return random.choice(classes), round(random.uniform(0.8, 0.99), 2)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("♻️ Garbage Classification")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user_role.upper()}")

    st.session_state.nav = st.sidebar.radio(
        "Navigation",
        ["User Profile", "Upload Garbage", "Reward History"]
    )

    if st.sidebar.button("Logout"):
        for k in st.session_state:
            st.session_state[k] = None
        st.rerun()

# =====================================================
# LOGIN / SIGNUP
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

    # ---------------- USER PROFILE ----------------
    if st.session_state.nav == "User Profile":
        st.title("👤 User Profile")
        st.write(f"**Email:** {st.session_state.user}")
        st.write("**Role:** User")
        st.info("Welcome to the Garbage Classification & Reward System")

    # ---------------- UPLOAD GARBAGE ----------------
    elif st.session_state.nav == "Upload Garbage":
        st.title("📷 Upload Garbage")

        st.subheader("🗂 Select Category")
        category = st.radio(
            "Choose category:",
            ["General Waste", "Furniture"],
            horizontal=True
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            st.image(uploaded_file, use_column_width=True)

            prediction, confidence = predict_garbage(category)

            st.subheader("🧠 Prediction Result")
            st.success(f"Predicted Class: {prediction}")
            st.info(f"Confidence: {confidence * 100:.0f}%")

            if st.button("🚚 Choose Delivery Station"):
                st.session_state.category = category
                st.session_state.page = "station"
                st.rerun()

        if st.session_state.page == "station":
            st.subheader("🚚 Delivery Station")

            station = st.radio(
                "Select station:",
                [
                    "Station A – City Recycling Center",
                    "Station B – Community Drop-off Point",
                    "Station C – Furniture Collection Hub"
                ]
            )

            st.success("🎁 Reward Preview: 10 points (Status: PENDING)")

            if st.button("✅ Confirm Station"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""
                    INSERT INTO rewards (user_email, category, station, points, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    st.session_state.user,
                    st.session_state.category,
                    station,
                    10,
                    "PENDING"
                ))
                conn.commit()
                conn.close()

                st.session_state.page = "upload"
                st.success("Delivery confirmed. Reward pending admin approval.")
                st.rerun()

    # ---------------- REWARD HISTORY ----------------
    elif st.session_state.nav == "Reward History":
        st.title("🎁 Reward History")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT category, station, points, status
            FROM rewards
            WHERE user_email=?
              AND category IS NOT NULL
              AND station IS NOT NULL
            ORDER BY id DESC
        """, (st.session_state.user,))
        rewards = c.fetchall()
        conn.close()

        if not rewards:
            st.info("No valid rewards yet")

        for cat, station, points, status in rewards:
            st.write(
                f"Category: {cat} | "
                f"Station: {station} | "
                f"Points: {points} | "
                f"Status: [{status}]"
            )

# =====================================================
# ADMIN DASHBOARD
# =====================================================
elif st.session_state.user_role == "admin":
    st.title("🛠 Admin Dashboard")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, user_email, category, station, points
        FROM rewards
        WHERE status='PENDING'
          AND category IS NOT NULL
          AND station IS NOT NULL
    """)
    rewards = c.fetchall()
    conn.close()

    if not rewards:
        st.info("No pending rewards")

    for rid, email, cat, station, points in rewards:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.write(email)
        col2.write(cat)
        col3.write(station)
        col4.write(points)
        col5.write("PENDING")

        if col6.button("Approve", key=f"approve_{rid}"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "UPDATE rewards SET status='APPROVED' WHERE id=?",
                (rid,)
            )
            conn.commit()
            conn.close()
            st.success("Reward approved!")
            st.rerun()

import streamlit as st
import sqlite3
import random
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------
# CONFIG
# -----------------------
DB_PATH = "garbage_app.db"
st.set_page_config(page_title="Garbage Classification System", layout="wide")

# -----------------------
# DATABASE INIT + MIGRATION
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # REWARDS TABLE (BASE)
    c.execute("""
    CREATE TABLE IF NOT EXISTS rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        points INTEGER,
        status TEXT
    )
    """)

    # ---- AUTO MIGRATION ----
    c.execute("PRAGMA table_info(rewards)")
    columns = [col[1] for col in c.fetchall()]

    if "category" not in columns:
        c.execute("ALTER TABLE rewards ADD COLUMN category TEXT")

    conn.commit()
    conn.close()

init_db()

# -----------------------
# SESSION STATE
# -----------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "login_type" not in st.session_state:
    st.session_state.login_type = "User"

# -----------------------
# AUTH FUNCTIONS
# -----------------------
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

# -----------------------
# SIMULATED AI PREDICTION
# -----------------------
def predict_garbage(category):
    if category == "General Waste":
        classes = ["Plastic", "Paper", "Metal", "Organic Waste"]
    else:
        classes = ["Chair", "Table", "Sofa", "Cabinet"]

    return random.choice(classes), round(random.uniform(0.75, 0.99), 2)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("♻️ Garbage Classification System")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user_role.upper()}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.user_role = None
        st.rerun()

# -----------------------
# LOGIN / SIGNUP
# -----------------------
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

# -----------------------
# USER DASHBOARD
# -----------------------
elif st.session_state.user_role == "user":
    st.title("👤 User Dashboard")

    st.subheader("🗂 Select Garbage Category")
    category = st.radio(
        "Choose category:",
        ["General Waste", "Furniture"],
        horizontal=True
    )

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

        prediction, confidence = predict_garbage(category)
        st.success(f"Predicted Class: {prediction}")
        st.info(f"Confidence: {confidence * 100:.0f}%")

        if st.button("Submit Garbage"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "INSERT INTO rewards (user_email, category, points, status) VALUES (?, ?, ?, ?)",
                (st.session_state.user, category, 10, "PENDING")
            )
            conn.commit()
            conn.close()
            st.success("Submitted! Pending admin approval.")
            st.rerun()

    # ---- REWARD STATUS (FIXED) ----
    st.subheader("🎁 Reward Status")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, category, points, status
        FROM rewards
        WHERE user_email=?
        ORDER BY id DESC
    """, (st.session_state.user,))
    rewards = c.fetchall()
    conn.close()

    if not rewards:
        st.info("No rewards yet")

    for rid, cat, points, status in rewards:
        st.write(f"{cat} | {points} points | {status}")

# -----------------------
# ADMIN DASHBOARD
# -----------------------
elif st.session_state.user_role == "admin":
    st.title("🛠 Admin Dashboard")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, user_email, category, points
        FROM rewards
        WHERE status='PENDING'
    """)
    rewards = c.fetchall()
    conn.close()

    if not rewards:
        st.info("No pending rewards")

    for rid, email, category, points in rewards:
        col1, col2, col3, col4 = st.columns(4)
        col1.write(email)
        col2.write(category)
        col3.write(points)

        if col4.button("Approve", key=rid):
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

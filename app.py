import streamlit as st
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------
# CONFIG
# -----------------------
DB_PATH = "garbage_app.db"

st.set_page_config(page_title="Garbage Classification System", layout="wide")

# -----------------------
# DATABASE INIT
# -----------------------
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
        points INTEGER,
        status TEXT
    )
    """)

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
# SIDEBAR
# -----------------------
st.sidebar.title("♻️ Garbage Classification")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user_role.upper()}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.user_role = None
        st.rerun()

# -----------------------
# LOGIN / SIGNUP PAGE
# -----------------------
if not st.session_state.user:
    st.title("🔐 Login / Signup")

    st.radio(
        "Login as:",
        ["User", "Admin"],
        horizontal=True,
        key="login_type"
    )

    role = st.session_state.login_type.lower()
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(email, password, role):
                st.session_state.user = email
                st.session_state.user_role = role
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        username = st.text_input("Username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            if signup_user(username, email, password, role):
                st.success("Account created. Please login.")
            else:
                st.error("Email already exists")

# -----------------------
# USER DASHBOARD
# -----------------------
elif st.session_state.user_role == "user":
    st.title("👤 User Dashboard")

    st.subheader("📷 Garbage Classification (Demo)")
    st.info("Pretend AI model classifies garbage here")

    if st.button("Submit Garbage"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO rewards (user_email, points, status) VALUES (?, ?, ?)",
            (st.session_state.user, 10, "PENDING")
        )
        conn.commit()
        conn.close()
        st.success("Garbage submitted! Reward pending admin approval.")

    st.subheader("🎁 Reward Status")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, points, status
        FROM rewards
        WHERE user_email=?
        ORDER BY id DESC
    """, (st.session_state.user,))
    rewards = c.fetchall()

    for rid, points, status in rewards:
        st.write(f"Points: {points} | Status: {status}")

        if status == "APPROVED":
            if st.button("🎉 Claim Reward", key=f"claim_{rid}"):
                c.execute("""
                    UPDATE rewards
                    SET status='EARNED'
                    WHERE id=?
                """, (rid,))
                conn.commit()
                st.success("Reward claimed!")
                st.rerun()

    conn.close()

# -----------------------
# ADMIN DASHBOARD
# -----------------------
elif st.session_state.user_role == "admin":
    st.title("🛠 Admin Dashboard")

    st.subheader("📋 Pending Reward Approvals")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, user_email, points, status
        FROM rewards
        WHERE status='PENDING'
    """)
    rewards = c.fetchall()

    if not rewards:
        st.info("No pending rewards")

    for rid, email, points, status in rewards:
        col1, col2, col3, col4 = st.columns(4)
        col1.write(email)
        col2.write(points)
        col3.write(status)

        if col4.button("Approve", key=f"approve_{rid}"):
            c.execute("""
                UPDATE rewards
                SET status='APPROVED'
                WHERE id=?
            """, (rid,))
            conn.commit()
            st.success("Reward approved")
            st.rerun()

    conn.close()

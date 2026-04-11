import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
import os
import gdown
import json
import pandas as pd
import plotly.express as px
from datetime import date
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =============================
# CUSTOM CSS (UI DESIGN)
# =============================
st.markdown("""
<style>
.nav-logo {
    font-size: 22px;
    font-weight: 800;
    color: #2fa4dc;
    letter-spacing: 1px;
    white-space: normal;
    line-height: 1.2;
}

.nav-active {
    color: #2fa4dc;
    font-weight: 700;
    font-size: 15px;
    padding-top: 8px;
    line-height: 1.2;
    text-align: center;
}

/* text-style buttons */
div[data-testid="stButton"] button[kind="tertiary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #6f7782 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    white-space: pre-line !important;
    text-align: center !important;
    line-height: 1.2 !important;
    padding: 8px 4px !important;
}
div[data-testid="stButton"] button[kind="tertiary"]:hover {
    background: transparent !important;
    color: #2fa4dc !important;
    border: none !important;
}

/* blue CTA button */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #4aa3df !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    padding: 10px 18px !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #3b94d1 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
    background-color:#7cb342;
    padding:10px;
    text-align:center;
    color:white;
    font-size:14px;
'>
    🌱 Smart Recycling System: Promoting sustainability and rewarding eco-friendly actions.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #f5f7fa;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8f5e9;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

/* Titles */
h1, h2, h3 {
    color: #2e7d32;
}


/* Info box */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.auth-title-center {
    text-align: center;
    color: white;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 8px;
}

.auth-subtitle-center {
    text-align: center;
    color: rgba(255,255,255,0.92);
    font-size: 18px;
    margin-bottom: 28px;
}

.auth-note-center {
    text-align: center;
    color: rgba(255,255,255,0.88);
    font-size: 15px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

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
    
    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)
    
    # Admin table
    c.execute("""
    CREATE TABLE IF NOT EXISTS staff (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        staff_id TEXT UNIQUE,
        name TEXT,
        email TEXT,
        password TEXT
    )
    """)
    
    # Pickup Requests table
    c.execute("""
    CREATE TABLE IF NOT EXISTS pickup_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        predicted_label TEXT,
        confidence REAL,
        address TEXT,
        pickup_date TEXT,
        pickup_time_slot TEXT,
        note TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Redemptions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS redemptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        reward_name TEXT,
        points_used INTEGER,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Rewards table
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
# LOAD MODELS
# =============================
@st.cache_resource
def load_garbage_model():
    model_path = Path(__file__).parent / "FYP_general_waste.h5"

    if not model_path.exists():
        raise FileNotFoundError(
            f"General waste model not found: {model_path}"
        )

    # Detect Git LFS pointer / wrong file / corrupted upload
    with open(model_path, "rb") as f:
        header = f.read(256)

    if header.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "FYP_general_waste.h5 in the repository is a Git LFS pointer, "
            "not the actual model file. Upload the real .h5 binary to the repo "
            "or use a direct download method."
        )

    if len(header) < 8:
        raise RuntimeError(
            "FYP_general_waste.h5 is too small or corrupted."
        )

    try:
        # compile=False is safer for inference-only deployment
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load general waste model from {model_path}. "
            f"The file may be corrupted, incomplete, or not a valid HDF5 model. "
            f"Original error: {e}"
        )
    
@st.cache_resource
def load_furniture_model():
    model_path = "bulky_classifier.keras"
    labels_path = "bulky_class_names.json"

    # download model if not exists
    if not os.path.exists(model_path):
        model_url = "18e-LMCD635Grz6LxlGlhVt20UL60lVCo"
        gdown.download(f"https://drive.google.com/uc?id={model_url}", model_path, quiet=False)

    # download labels if not exists
    if not os.path.exists(labels_path):
        labels_url = "1OAb0O3nvgiB6q_Xn_8naU4oVGkPFrYwd"
        gdown.download(f"https://drive.google.com/uc?id={labels_url}", labels_path, quiet=False)

    model = tf.keras.models.load_model(model_path)

    with open(labels_path, "r") as f:
        class_names = json.load(f)

    return model, class_names
# =============================
# SESSION STATE DEFAULTS
# =============================
for key in ["role", "user", "category", "reward_pending", "show_reward", "page"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "show_forgot_password" not in st.session_state:
    st.session_state.show_forgot_password = False
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

def signup_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    if c.fetchone():
        conn.close()
        return False
    hashed_password = generate_password_hash(password)
    c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
              (name, email, hashed_password))
    conn.commit()
    conn.close()
    return True

def reset_user_password(email, new_password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    user = c.fetchone()

    if not user:
        conn.close()
        return False

    hashed_password = generate_password_hash(new_password)
    c.execute("UPDATE users SET password=? WHERE email=?", (hashed_password, email))
    conn.commit()
    conn.close()
    return True

def goto_user_page(page_name):
    st.session_state.page = page_name
    st.rerun()
# =============================
# ROLE SELECTION (FIXED)
# =============================
if st.session_state.role is None:

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
                          url("https://images.unsplash.com/photo-1621451537084-482c73073a0f");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .role-title {
        text-align: center;
        color: white;
        font-size: 54px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .role-subtitle {
        text-align: center;
        color: white;
        font-size: 22px;
        margin-bottom: 30px;
    }

    .role-box {
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 18px;
        padding: 40px 30px;
        max-width: 900px;
        margin: 0 auto;
    }

    .role-note {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 16px;
        margin-bottom: 25px;
    }

    div[data-testid="stButton"] button {
        height: 55px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        background-color: #43a047 !important;
        color: white !important;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #2e7d32 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:18vh;'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 2.2, 1])

    with center:
        st.markdown('<div class="role-box">', unsafe_allow_html=True)
        st.markdown('<div class="role-title">♻️ Smart Recycling System</div>', unsafe_allow_html=True)
        st.markdown('<div class="role-subtitle">Promoting sustainability and rewarding eco-friendly actions</div>', unsafe_allow_html=True)
        st.markdown('<div class="role-note">Choose your role to continue</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👤 USER", use_container_width=True, key="landing_user"):
                st.session_state.role = "USER"
                st.rerun()

        with col2:
            if st.button("🛠 ADMIN", use_container_width=True, key="landing_admin"):
                st.session_state.role = "ADMIN"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# ADMIN LOGIN / SIGNUP
# =============================
if st.session_state.role == "ADMIN" and st.session_state.user is None:
    
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                          url("https://images.unsplash.com/photo-1621451537084-482c73073a0f");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .auth-title-center {
        text-align: center;
        color: white;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .auth-subtitle-center {
        text-align: center;
        color: rgba(255,255,255,0.92);
        font-size: 18px;
        margin-bottom: 28px;
    }

    .auth-note-center {
        text-align: center;
        color: rgba(255,255,255,0.88);
        font-size: 15px;
        margin-bottom: 18px;
    }

    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span {
        color: white !important;
        font-weight: 600 !important;
    }

    div[role="radiogroup"] label,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        color: white !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stTextInput"] input {
        background-color: rgba(255,255,255,0.95) !important;
        color: black !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12vh;'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1.2, 2.2, 1.2])

    with center:
        st.markdown('<div class="auth-title-center">🛠 ADMIN Login / Sign Up</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle-center">Manage rewards, pickups, approvals, and analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-note-center">Choose an option to continue</div>', unsafe_allow_html=True)

        if st.button("← Back to Role Selection", key="back_from_admin"):
            st.session_state.role = None
            st.rerun()

        st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Choose an option</p>", unsafe_allow_html=True)
        option = st.radio("", ["Login", "Sign Up"], key="admin_option", label_visibility="collapsed")

        if option == "Login":
            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Staff ID</p>", unsafe_allow_html=True)
            staff_id = st.text_input("", key="admin_login_id", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Password</p>", unsafe_allow_html=True)
            password = st.text_input("", type="password", key="admin_login_pw", label_visibility="collapsed")

            if st.button("Login", key="admin_login_btn"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT password FROM staff WHERE staff_id=?", (staff_id,))
                row = c.fetchone()
                conn.close()

                if row and check_password_hash(row[0], password):
                    st.session_state.user = staff_id
                    st.session_state.page = "Home"
                    st.success("ADMIN login successful!")
                    st.rerun()
                else:
                    st.error("Invalid StaffID or Password")

        elif option == "Sign Up":
            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Staff ID</p>", unsafe_allow_html=True)
            staff_id = st.text_input("", key="admin_signup_id", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Name</p>", unsafe_allow_html=True)
            name = st.text_input("", key="admin_signup_name", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Email</p>", unsafe_allow_html=True)
            email = st.text_input("", key="admin_signup_email", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Password</p>", unsafe_allow_html=True)
            password = st.text_input("", type="password", key="admin_signup_pw", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Confirm Password</p>", unsafe_allow_html=True)
            confirm = st.text_input("", type="password", key="admin_signup_confirm", label_visibility="collapsed")

            if st.button("Sign Up", key="admin_signup_btn"):
                if password != confirm:
                    st.error("Passwords do not match")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT * FROM staff WHERE staff_id=?", (staff_id,))
                    if c.fetchone():
                        st.error("StaffID already exists")
                    else:
                        hashed = generate_password_hash(password)
                        c.execute(
                            "INSERT INTO staff (staff_id, name, email, password) VALUES (?,?,?,?)",
                            (staff_id, name, email, hashed)
                        )
                        conn.commit()
                        conn.close()
                        st.success("Admin Sign Up successful! Please login.")

# =============================
# USER LOGIN / SIGNUP
# =============================
elif st.session_state.role == "USER" and st.session_state.user is None:
    
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                          url("https://images.unsplash.com/photo-1621451537084-482c73073a0f");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .auth-title-center {
        text-align: center;
        color: white;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .auth-subtitle-center {
        text-align: center;
        color: rgba(255,255,255,0.92);
        font-size: 18px;
        margin-bottom: 28px;
    }

    .auth-note-center {
        text-align: center;
        color: rgba(255,255,255,0.88);
        font-size: 15px;
        margin-bottom: 18px;
    }

    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span {
        color: white !important;
        font-weight: 600 !important;
    }

    div[role="radiogroup"] label,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {
        color: white !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: rgba(255,255,255,0.95) !important;
        color: black !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12vh;'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1.2, 2.2, 1.2])

    with center:
        st.markdown('<div class="auth-title-center">👤 USER Login / Sign Up</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle-center">Access your recycling account and continue your sustainability journey</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-note-center">Choose an option to continue</div>', unsafe_allow_html=True)

        if st.button("← Back to Role Selection", key="back_from_user"):
            st.session_state.role = None
            st.rerun()

        st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Choose an option</p>", unsafe_allow_html=True)
        option = st.radio("", ["Login", "Sign Up"], key="user_option", label_visibility="collapsed")
        
        if option == "Login":
            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Email</p>", unsafe_allow_html=True)
            email = st.text_input("", key="user_login_email", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Password</p>", unsafe_allow_html=True)
            password = st.text_input("", type="password", key="user_login_pw", label_visibility="collapsed")

            if st.button("Login", key="user_login_btn"):
                if login_user(email, password):
                    st.session_state.user = email
                    st.session_state.page = "Home"
                    st.success("USER login successful!")
                    st.rerun()
                else:
                    st.error("Invalid Email or Password")
                    
            if st.button("Forgot Password?", key="forgot_password_btn"):
                    st.session_state.show_forgot_password = True

            if st.session_state.show_forgot_password:
                st.markdown('<p style="color:white; font-size:22px; font-weight:700;">🔑 Reset Password</p>', unsafe_allow_html=True)

                st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Enter your registered email</p>", unsafe_allow_html=True)
                reset_email = st.text_input("", key="reset_email", label_visibility="collapsed")
                st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Enter new password</p>", unsafe_allow_html=True)
                new_password = st.text_input("", type="password", key="reset_new_password", label_visibility="collapsed")

                # Confirm Password
                st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Confirm new password</p>", unsafe_allow_html=True)
                confirm_new_password = st.text_input("", type="password", key="reset_confirm_password", label_visibility="collapsed")


                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Update Password", key="update_password_btn"):
                        if not reset_email or not new_password or not confirm_new_password:
                            st.error("All fields are required.")
                        elif new_password != confirm_new_password:
                            st.error("Passwords do not match.")
                        elif reset_user_password(reset_email, new_password):
                            st.success("Password updated successfully. Please log in.")
                            st.session_state.show_forgot_password = False
                        else:
                            st.error("Email not found.")

                with col2:
                    if st.button("Cancel", key="cancel_reset_btn"):
                        st.session_state.show_forgot_password = False
                        st.rerun()

        elif option == "Sign Up":
            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Name</p>", unsafe_allow_html=True)
            name = st.text_input("", key="user_signup_name", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Email</p>", unsafe_allow_html=True)
            email = st.text_input("", key="user_signup_email", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Password</p>", unsafe_allow_html=True)
            password = st.text_input("", type="password", key="user_signup_pw", label_visibility="collapsed")

            st.markdown("<p style='color:white; font-weight:600; margin-bottom:6px;'>Confirm Password</p>", unsafe_allow_html=True)
            confirm = st.text_input("", type="password", key="user_signup_confirm", label_visibility="collapsed")

            if st.button("Sign Up", key="user_signup_btn"):
                if password != confirm:
                    st.error("Passwords do not match")
                elif not name or not email or not password:
                    st.error("All fields are required")
                elif signup_user(name, email, password):
                    st.success("Sign Up successful! Please login.")
                else:
                    st.error("Email already registered")
# =============================
# LOGOUT HANDLER
# =============================
if st.session_state.page == "Logout":
    for key in ["role", "user", "category", "reward_pending", "show_reward", "page"]:
        st.session_state[key] = None
    st.rerun()

# =============================
# ADMIN DASHBOARD
# =============================
elif st.session_state.role == "ADMIN" and st.session_state.user:
    # =============================
    # ADMIN TOP NAVBAR
    # =============================
    nav1, nav2, nav3, nav4, nav5, nav6, nav7 = st.columns([3,1.2,1.8,1.8,2.5,2.0,1.0])

    with nav1:
        st.markdown('<div class="nav-logo">♻️ SMART<br>RECYCLING</div>', unsafe_allow_html=True)

    # HOME
    with nav2:
        if st.session_state.page == "Home":
            st.markdown('<div class="nav-active">HOME</div>', unsafe_allow_html=True)
        else:
            if st.button("HOME", key="admin_home", type="tertiary", use_container_width=True):
                st.session_state.page = "Home"
                st.rerun()

    # PENDING REWARDS
    with nav3:
        if st.session_state.page == "Pending Rewards":
            st.markdown('<div class="nav-active">PENDING<br>REWARDS</div>', unsafe_allow_html=True)
        else:
            if st.button("PENDING\nREWARDS", key="admin_pending", type="tertiary", use_container_width=True):
                st.session_state.page = "Pending Rewards"
                st.rerun()

    # PICKUP REQUESTS
    with nav4:
        if st.session_state.page == "Pickup Requests":
            st.markdown('<div class="nav-active">PICKUP<br>REQUESTS</div>', unsafe_allow_html=True)
        else:
            if st.button("PICKUP\nREQUESTS", key="admin_requests", type="tertiary", use_container_width=True):
                st.session_state.page = "Pickup Requests"
                st.rerun()

    # SCHEDULED PICKUPS
    with nav5:
        if st.session_state.page == "Scheduled Pickups":
            st.markdown('<div class="nav-active">SCHEDULED<br>PICKUPS</div>', unsafe_allow_html=True)
        else:
            if st.button("SCHEDULED\nPICKUPS", key="admin_scheduled", type="tertiary", use_container_width=True):
                st.session_state.page = "Scheduled Pickups"
                st.rerun()

    # ANALYTICS
    with nav6:
        if st.session_state.page == "Analytics":
            st.markdown('<div class="nav-active">ANALYTICS</div>', unsafe_allow_html=True)
        else:
            if st.button("ANALYTICS", key="admin_analytics", type="tertiary", use_container_width=True):
                st.session_state.page = "Analytics"
                st.rerun()

    # LOGOUT
    with nav7:
        if st.button("LOG\nOUT", key="admin_logout", type="tertiary", use_container_width=True):
            for key in ["role", "user", "category", "reward_pending", "show_reward", "page"]:
                st.session_state[key] = None
            st.rerun()

    st.markdown("<hr style='margin-top:10px; margin-bottom:20px;'>", unsafe_allow_html=True)
    if st.session_state.page == "Home":
        # =============================
        # HERO BANNER 
        # =============================
        st.markdown("""
        <div style="
            background-image: url('https://images.unsplash.com/photo-1604187351574-c75ca79f5807');
            height: 300px;
            background-size: cover;
            background-position: center;
            border-radius: 12px;
            position: relative;
            margin-bottom: 20px;
        ">
            <div style="
                position: absolute;
                bottom: 20px;
                left: 30px;
                color: white;
            ">
                <h1 style="margin-bottom:5px;">♻️ Smart Recycling Reward System</h1>
                <p style="font-size:18px;">Earn rewards while saving the environment</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # spacing
        st.markdown("<br>", unsafe_allow_html=True)      

        st.title("Admin Dashboard")
        st.write("Welcome, Admin.")
        st.write("Use the top navigation bar to manage the system.")

    elif st.session_state.page == "Pending Rewards":
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
                    st.rerun()
        else:
            st.info("No pending rewards.")
            
    elif st.session_state.page == "Pickup Requests":
        st.title("Admin Dashboard - Pickup Requests")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, user_email, predicted_label, confidence, status, created_at
            FROM pickup_requests
            WHERE status = 'PENDING_APPROVAL'
            ORDER BY created_at DESC
        """)
        pickup_requests = c.fetchall()
        conn.close()

        if pickup_requests:
            for request in pickup_requests:
                request_id, user_email, predicted_label, confidence, status, created_at = request

                st.write(f"**Request ID:** {request_id}")
                st.write(f"**User:** {user_email}")
                st.write(f"**Predicted Item:** {predicted_label}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.write(f"**Status:** {status}")
                st.write(f"**Created At:** {created_at}")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(f"APPROVE PICKUP {request_id}", key=f"approve_pickup_{request_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            "UPDATE pickup_requests SET status='APPROVED' WHERE id=?",
                            (request_id,)
                        )
                        conn.commit()
                        conn.close()
                        st.success(f"Pickup request {request_id} approved!")
                        st.rerun()

                with col2:
                    if st.button(f"REJECT PICKUP {request_id}", key=f"reject_pickup_{request_id}"):
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            "UPDATE pickup_requests SET status='REJECTED' WHERE id=?",
                            (request_id,)
                        )
                        conn.commit()
                        conn.close()
                        st.warning(f"Pickup request {request_id} rejected.")
                        st.rerun()

                st.markdown("---")
        else:
            st.info("No pending pickup requests.")
            
    elif st.session_state.page == "Scheduled Pickups":
        st.title("Admin Dashboard - Scheduled Pickups")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, user_email, predicted_label, address, pickup_date, pickup_time_slot, note, status
            FROM pickup_requests
            WHERE status = 'SCHEDULED'
            ORDER BY pickup_date ASC
        """)
        scheduled_pickups = c.fetchall()
        conn.close()

        if scheduled_pickups:
            for pickup in scheduled_pickups:
                request_id, user_email, predicted_label, address, pickup_date, pickup_time_slot, note, status = pickup

                st.write(f"**Request ID:** {request_id}")
                st.write(f"**User:** {user_email}")
                st.write(f"**Item:** {predicted_label}")
                st.write(f"**Address:** {address}")
                st.write(f"**Pickup Date:** {pickup_date}")
                st.write(f"**Time Slot:** {pickup_time_slot}")
                st.write(f"**Note:** {note if note else 'No additional note'}")
                st.write(f"**Status:** {status}")

                if st.button(f"MARK COMPLETED {request_id}", key=f"complete_pickup_{request_id}"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()

                    # Mark pickup request as completed
                    c.execute(
                        "UPDATE pickup_requests SET status='COMPLETED' WHERE id=?",
                        (request_id,)
                    )

                    # Add 30 points to rewards table after completion
                    c.execute(
                        "INSERT INTO rewards (user_email, points, status, station) VALUES (?, ?, ?, ?)",
                        (user_email, 30, "APPROVED", "Door-to-door pickup completed")
                    )

                    conn.commit()
                    conn.close()

                    st.success(f"Pickup request {request_id} marked as completed. 30 points added to user.")
                    st.rerun()

                st.markdown("---")
        else:
            st.info("No scheduled pickups.")
            
    elif st.session_state.page == "Analytics":
        st.title("Admin Dashboard - Analytics")
        st.write("Monitor users, pickup activity, rewards, and redemption performance.")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Summary numbers
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM rewards")
        total_rewards = c.fetchone()[0]

        c.execute("SELECT COALESCE(SUM(points), 0) FROM rewards")
        total_points_awarded = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM pickup_requests")
        total_pickup_requests = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='PENDING_APPROVAL'")
        pending_pickups = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='APPROVED'")
        approved_pickups = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='SCHEDULED'")
        scheduled_pickups = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='COMPLETED'")
        completed_pickups = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE status='REJECTED'")
        rejected_pickups = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM redemptions")
        total_redemptions = c.fetchone()[0]

        c.execute("SELECT COALESCE(SUM(points_used), 0) FROM redemptions")
        total_points_redeemed = c.fetchone()[0]

        conn.close()

    # =============================
    # TOP SUMMARY CARDS
    # =============================
        top1, top2, top3, top4 = st.columns(4)

        with top1:
            st.metric("Total Users", total_users)

        with top2:
            st.metric("Pickup Requests", total_pickup_requests)

        with top3:
            st.metric("Points Awarded", total_points_awarded)

        with top4:
            st.metric("Redemptions", total_redemptions)

        st.markdown("<br>", unsafe_allow_html=True)

        mid1, mid2, mid3, mid4 = st.columns(4)

        with mid1:
            st.metric("Pending", pending_pickups)

        with mid2:
            st.metric("Approved", approved_pickups)

        with mid3:
            st.metric("Scheduled", scheduled_pickups)

        with mid4:
            st.metric("Completed", completed_pickups)

        st.markdown("<br>", unsafe_allow_html=True)

    # =============================
    # CHART 1: PICKUP STATUS DISTRIBUTION
    # =============================
        status_data = pd.DataFrame({
            "Status": ["Pending", "Approved", "Scheduled", "Completed", "Rejected"],
            "Count": [pending_pickups, approved_pickups, scheduled_pickups, completed_pickups, rejected_pickups]
        })

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_status = px.pie(
                status_data,
                names="Status",
                values="Count",
                title="Pickup Status Distribution"
            )
            fig_status.update_layout(height=420)
            st.plotly_chart(fig_status, use_container_width=True)

    # =============================
    # CHART 2: POINTS OVERVIEW
    # =============================
        points_data = pd.DataFrame({
            "Category": ["Points Awarded", "Points Redeemed"],
            "Value": [total_points_awarded, total_points_redeemed]
        })

        with chart_col2:
            fig_points = px.bar(
                points_data,
                x="Category",
                y="Value",
                title="Points Overview",
                text="Value"
            )
            fig_points.update_layout(height=420)
            st.plotly_chart(fig_points, use_container_width=True)

    # =============================
    # CHART 3: PICKUP REQUESTS OVER TIME
    # =============================
        conn = sqlite3.connect(DB_PATH)
        requests_over_time = pd.read_sql_query("""
            SELECT DATE(created_at) AS request_date, COUNT(*) AS total_requests
            FROM pickup_requests
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
        """, conn)
        conn.close()

        st.markdown("<br>", unsafe_allow_html=True)

        if not requests_over_time.empty:
            fig_line = px.line(
                requests_over_time,
                x="request_date",
                y="total_requests",
                markers=True,
                title="Pickup Requests Over Time"
            )
            fig_line.update_layout(height=420)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No pickup request data available for trend chart.")

    # =============================
    # OPTIONAL EXTRA DETAILS
    # =============================
        st.markdown("<br>", unsafe_allow_html=True)
        extra1, extra2 = st.columns(2)

        with extra1:
            st.metric("Rejected Pickup Requests", rejected_pickups)

        with extra2:
            st.metric("Total Rewards Records", total_rewards)
        
# =============================
# USER FLOW
# =============================
elif st.session_state.role == "USER" and st.session_state.user:

    # TOP NAVBAR FOR ALL USER PAGES
    nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([3.2, 1.2, 2.0, 2.4, 2.2, 1.6])

    with nav1:
        st.markdown('<div class="nav-logo">♻️ SMART<br>RECYCLING</div>', unsafe_allow_html=True)

    with nav2:
        if st.session_state.page == "Home":
            st.markdown('<div class="nav-active">HOME</div>', unsafe_allow_html=True)
        else:
            if st.button("HOME", key="nav_home", type="tertiary", use_container_width=True):
                goto_user_page("Home")

    with nav3:
        if st.session_state.page == "Reward History":
            st.markdown('<div class="nav-active">REWARD<br>HISTORY</div>', unsafe_allow_html=True)
        else:
            if st.button("REWARD\nHISTORY", key="nav_reward_history", type="tertiary", use_container_width=True):
                goto_user_page("Reward History")

    with nav4:
        if st.session_state.page == "Redemption History":
            st.markdown('<div class="nav-active">REDEMPTION<br>HISTORY</div>', unsafe_allow_html=True)
        else:
            if st.button("REDEMPTION\nHISTORY", key="nav_redemption_history", type="tertiary", use_container_width=True):
                goto_user_page("Redemption History")

    with nav5:
        if st.button("LOG\nOUT", key="nav_logout", type="tertiary", use_container_width=True):
            for key in ["role", "user", "category", "reward_pending", "show_reward", "page"]:
                st.session_state[key] = None
            st.rerun()
                
    with nav6:
        if st.session_state.page == "Profile":
            st.markdown('<div class="nav-active">PROFILE</div>', unsafe_allow_html=True)
        else:
            if st.button("PROFILE", key="nav_profile", type="tertiary", use_container_width=True):
                goto_user_page("Profile")

    st.markdown("<hr style='margin-top:10px; margin-bottom:20px;'>", unsafe_allow_html=True)
    
    if st.session_state.page == "Home":
        st.markdown("""
        <div style="
            background-image: url('https://images.unsplash.com/photo-1604187351574-c75ca79f5807');
            height: 300px;
            background-size: cover;
            background-position: center;
            border-radius: 12px;
            position: relative;
            margin-bottom: 20px;
        ">
            <div style="
                position: absolute;
                bottom: 20px;
                left: 30px;
                color: white;
            ">
                <h1 style="margin-bottom:5px;">♻️ Smart Recycling Reward System</h1>
                <p style="font-size:18px;">Earn rewards while saving the environment</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Welcome to your recycling dashboard")

        # your Home dashboard code continues here

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📤 Upload Waste", use_container_width=True):
                st.session_state.page = "Upload Waste"
                st.rerun()

        with col2:
            if st.button("🎁 Reward Status", use_container_width=True):
                st.session_state.page = "Reward Status"
                st.rerun()

        with col3:
            if st.button("🚚 Pickup Scheduling", use_container_width=True):
                st.session_state.page = "Pickup Scheduling"
                st.rerun()

        with col4:
            if st.button("🛒 Redeem Rewards", use_container_width=True):
                st.session_state.page = "Redeem Rewards"
                st.rerun()

    elif st.session_state.page == "Upload Waste" and st.session_state.category is None:
        st.markdown("## Select Category")
        st.markdown("**Step 1 of 3** · Choose the type of waste you want to classify")

        st.write(
            "Select the most suitable category below before continuing to image upload."
        )

        card1, card2 = st.columns(2)

        with card1:
            st.markdown("""
            <div style="
                background-color: white;
                border: 1px solid #d9e2ec;
                border-radius: 14px;
                padding: 18px;
                min-height: 180px;
                box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
            ">
                <h3 style="margin-top:0;">♻️ General Waste</h3>
                <p>Use this for everyday waste items such as paper, plastic, glass, metal, and trash.</p>
                <p><strong>Flow:</strong> Upload image → AI classification → reward processing</p>
            </div>
            """, unsafe_allow_html=True)

        with card2:
            st.markdown("""
            <div style="
                background-color: white;
                border: 1px solid #d9e2ec;
                border-radius: 14px;
                padding: 18px;
                min-height: 180px;
                box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
            ">
                <h3 style="margin-top:0;">🪑 Furniture</h3>
                <p>Use this for bulky household items such as chairs, tables, fridges, TVs, and wardrobes.</p>
                <p><strong>Flow:</strong> Upload image → approval → pickup scheduling</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        category = st.radio(
            "Choose waste type",
            ["General Waste", "Furniture"],
            horizontal=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        btn_left, btn_center, btn_right = st.columns([1, 1, 1])

        with btn_center:
            if st.button("Continue", use_container_width=True):
                st.session_state.category = category
                st.rerun()

    elif st.session_state.page == "Upload Waste" and st.session_state.reward_pending is None:
        st.subheader("Upload Image")

        if st.button("Change Waste Category"):
            st.session_state.category = None
            st.rerun()

        expected_furniture = None
        if st.session_state.category == "Furniture":
            st.info("Supported bulky categories: Chair, Fridge, Table, TV, Wardrobe")
            expected_furniture = st.selectbox(
                "Select the bulky item type you are uploading",
                ["chair image", "fridge image", "table image", "tv image", "wardrobe image"]
            )

        file = st.file_uploader("Upload garbage image", type=["jpg", "png", "jpeg"])

        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)

            if st.session_state.category == "General Waste":
                model = load_garbage_model()
                labels = ["Paper", "Plastic", "Metal", "Glass", "Cardboard", "Trash"]
            else:
                model, labels = load_furniture_model()

            img = image.resize((224, 224))
            img_array = np.array(img)

            if st.session_state.category == "General Waste":
                arr = np.expand_dims(img_array / 255.0, axis=0)
            else:
                arr = np.expand_dims(img_array, axis=0)
                arr = preprocess_input(arr.astype(np.float32))

            try:
                pred = model.predict(arr, verbose=0)
                confidence = float(np.max(pred))
                index = np.argmax(pred)

                if index >= len(labels):
                    st.error("Prediction error: label mismatch")
                else:
                    result = labels[index]

                    st.success(f"Prediction Result: {result}")
                    st.info(f"Confidence: {confidence * 100:.2f}%")

                    # =============================
                    # GENERAL WASTE LOGIC
                    # =============================
                    if st.session_state.category == "General Waste":
                        if confidence >= 0.70:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute(
                                "INSERT INTO rewards (user_email, points, status, station) VALUES (?, ?, ?, ?)",
                                (st.session_state.user, 10, "PENDING", None)
                            )
                            conn.commit()
                            conn.close()

                            st.session_state.reward_pending = True
                            st.success("✅ 10 points added successfully.")

                            if st.button("Check Reward"):
                                st.session_state.show_reward = True
                                st.rerun()
                        else:
                            st.warning("⚠️ Low confidence. No reward given. Please try another image.")

                    # =============================
                    # BULKY WASTE LOGIC
                    # =============================
                    elif st.session_state.category == "Furniture":
                        if confidence < 0.85:
                            st.warning("⚠️ Low confidence for bulky item. No pickup request created. Please try another image.")
                        elif result != expected_furniture:
                            st.error("Unsupported bulky item or mismatched prediction. No pickup request created.")
                        else:
                            conn = sqlite3.connect(DB_PATH)
                            c = conn.cursor()
                            c.execute(
                                """
                                INSERT INTO pickup_requests
                                (user_email, predicted_label, confidence, address, pickup_date, pickup_time_slot, note, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                (st.session_state.user, result, confidence, None, None, None, None, "PENDING_APPROVAL")
                                )
                            conn.commit()
                            conn.close()

                            st.success("✅ Bulky waste request submitted successfully.")
                            st.info("Your bulky waste request is now pending admin approval.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif st.session_state.page == "Reward Status":
        st.subheader("🎁 Reward Status")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT predicted_label, status
            FROM pickup_requests
            WHERE user_email=?
            ORDER BY created_at DESC
        """, (st.session_state.user,))
    
        requests = c.fetchall()
        conn.close()

        if requests:
            pending = []
            approved = []
            rejected = []

            for label, status in requests:
                if status == "PENDING_APPROVAL":
                    pending.append(label)
                elif status == "APPROVED":
                    approved.append(label)
                elif status == "REJECTED":
                    rejected.append(label)

            # Pending
            if pending:
                st.warning("⏳ Waiting for ADMIN approval:")
                for item in pending:
                    st.write(f"- {item}")

            # Approved
            if approved:
                st.success("✅ Approved items:")
                for item in approved:
                    st.write(f"- {item}")

            # Rejected
            if rejected:
                st.error("❌ Rejected items:")
                for item in rejected:
                    st.write(f"- {item}")

        else:
            st.info("No uploaded items yet.")
            
        st.markdown("### Suggested Recycling Stations")

        stations = [
            {
                "name": "1Recycling Centre (1RC) @ 1 Utama",
                "address": "City Centre, B2 Highstreet, 1, Lebuh Bandar Utama, Petaling Jaya",
                "hours": "Open · Closes 10 PM"
            },
            {
                "name": "IPC Recycling & Buy-Back Centre",
                "address": "IPC Shopping Centre, Ladies Parking, Level P1, Petaling Jaya",
                "hours": "Open · Closes 10 PM"
            },
            {
                "name": "PJ Eco Recycling Plaza",
                "address": "Jalan SS8/39, Petaling Jaya",
                "hours": "Open · Closes 4:30 PM"
            }
        ]

        for s in stations:
            st.markdown(f"**{s['name']}**")
            st.write(s["address"])
            st.write(s["hours"])
            st.markdown("---")
            
    elif st.session_state.page == "Pickup Scheduling":
        st.subheader("🚚 Pickup Scheduling")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, predicted_label, status
            FROM pickup_requests
            WHERE user_email=? AND status='APPROVED'
            ORDER BY created_at DESC
        """, (st.session_state.user,))
        approved_requests = c.fetchall()
        conn.close()

        if approved_requests:
            st.info(f"Approved pickup items: {len(approved_requests)}")
            st.success("Your pickup request(s) have been approved. Please select one item to schedule.")

            st.markdown("### Items approved by admin")
            for _, label, _ in approved_requests:
                st.write(f"- {label}")

            item_options = {
                f"{label} (Request ID: {request_id})": request_id
                for request_id, label, _ in approved_requests
            }

            selected_item = st.selectbox(
                "Select approved item to schedule",
                list(item_options.keys())
            )

            selected_request_id = item_options[selected_item]

            address = st.text_area("Pickup Address")
            pickup_date = st.date_input("Select Pickup Date", min_value=date.today())
            pickup_time_slot = st.selectbox(
                "Select Pickup Time Slot",
                ["9:00 AM - 12:00 PM", "12:00 PM - 3:00 PM", "3:00 PM - 6:00 PM"]
            )
            note = st.text_area("Additional Note (Optional)")

            if st.button("Confirm Pickup Schedule"):
                if pickup_date < date.today():
                    st.error("Past dates are not allowed. Please select today or a future date.")
                elif not address.strip():
                    st.error("Please enter your pickup address.")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        UPDATE pickup_requests
                        SET address=?, pickup_date=?, pickup_time_slot=?, note=?, status='SCHEDULED'
                        WHERE id=?
                    """, (address, str(pickup_date), pickup_time_slot, note, selected_request_id))
                    conn.commit()
                    conn.close()

                    st.success("✅ Selected pickup item scheduled successfully!")
                    st.rerun()
        else:
            st.info("No approved pickup request available for scheduling yet.")
            
    elif st.session_state.page == "Reward History":
        st.subheader("📜 Reward History")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, points, status, station
            FROM rewards
            WHERE user_email=?
            ORDER BY id DESC
        """, (st.session_state.user,))
        reward_history = c.fetchall()
        conn.close()

        if reward_history:
            total_points = sum(record[1] for record in reward_history)
            st.info(f"Total Points Collected: {total_points}")

            for reward in reward_history:
                reward_id, points, status, station = reward

                st.write(f"**Reward ID:** {reward_id}")
                st.write(f"**Points:** {points}")
                st.write(f"**Status:** {status}")
                st.write(f"**Source / Station:** {station if station else 'Not assigned yet'}")
                st.markdown("---")
        else:
            st.info("No reward history found yet.")
            
    elif st.session_state.page == "Redeem Rewards":
        st.subheader("🎁 Redeem Rewards")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Total earned points
        c.execute("SELECT COALESCE(SUM(points), 0) FROM rewards WHERE user_email=?", (st.session_state.user,))
        total_earned = c.fetchone()[0]

        # Total redeemed points
        c.execute("SELECT COALESCE(SUM(points_used), 0) FROM redemptions WHERE user_email=?", (st.session_state.user,))
        total_redeemed = c.fetchone()[0]

        conn.close()

        available_points = total_earned - total_redeemed

        st.info(f"Available Points: {available_points}")

        reward_catalog = [
            ("💳 TNG Reload Pin RM5", 50),
            ("🛍️ AEON Voucher RM15", 150),
            ("🛒 Shopee Voucher RM20", 200),
            ("🍔 GrabFood Voucher RM10", 100),
            ("📦 Lazada Voucher RM20", 200)
        ]

        for reward_name, points_required in reward_catalog:
            st.subheader(reward_name)
            st.write(f"Required Points: {points_required}")

            progress_value = min(available_points / points_required, 1.0)
            st.progress(progress_value)

            points_needed = points_required - available_points

            if available_points >= points_required:
                st.success("🎉 You can redeem this reward now!")

                if st.button(f"Redeem {reward_name}", key=f"redeem_{reward_name}"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute(
                        """
                        INSERT INTO redemptions (user_email, reward_name, points_used, status)
                        VALUES (?, ?, ?, ?)
                        """,
                        (st.session_state.user, reward_name, points_required, "COMPLETED")
                    )
                    conn.commit()
                    conn.close()

                    st.success(f"✅ You have successfully redeemed {reward_name}.")
                    st.rerun()
            else:
                st.button(f"Redeem {reward_name}", key=f"redeem_{reward_name}", disabled=True)
                st.caption(f"You need {points_needed} more points to redeem this reward.")

            st.divider()

    elif st.session_state.page == "Redemption History":
        st.subheader("🧾 Redemption History")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT reward_name, points_used, status, created_at
            FROM redemptions
            WHERE user_email=?
            ORDER BY created_at DESC
        """, (st.session_state.user,))
        redemption_history = c.fetchall()
        conn.close()

        if redemption_history:
            total_spent = sum(record[1] for record in redemption_history)
            st.info(f"Total Points Spent: {total_spent}")

            for record in redemption_history:
                reward_name, points_used, status, created_at = record

                st.write(f"**Reward:** {reward_name}")
                st.write(f"**Points Used:** {points_used}")
                st.write(f"**Status:** {status}")
                st.write(f"**Redeemed At:** {created_at}")
                st.markdown("---")
        else:
            st.info("No redemption history found yet.")
            
    elif st.session_state.page == "Profile":
        st.title("Profile")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # user info
        c.execute("SELECT name, email FROM users WHERE email=?", (st.session_state.user,))
        user_info = c.fetchone()

        # total earned points
        c.execute("SELECT COALESCE(SUM(points), 0) FROM rewards WHERE user_email=?", (st.session_state.user,))
        total_earned = c.fetchone()[0]

        # total redeemed points
        c.execute("SELECT COALESCE(SUM(points_used), 0) FROM redemptions WHERE user_email=?", (st.session_state.user,))
        total_redeemed = c.fetchone()[0]

        # total reward records
        c.execute("SELECT COUNT(*) FROM rewards WHERE user_email=?", (st.session_state.user,))
        total_reward_records = c.fetchone()[0]

        # total pickup requests
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE user_email=?", (st.session_state.user,))
        total_pickup_requests = c.fetchone()[0]

        # completed pickups
        c.execute("SELECT COUNT(*) FROM pickup_requests WHERE user_email=? AND status='COMPLETED'", (st.session_state.user,))
        completed_pickups = c.fetchone()[0]

        # total redemptions
        c.execute("SELECT COUNT(*) FROM redemptions WHERE user_email=?", (st.session_state.user,))
        total_redemptions = c.fetchone()[0]

        conn.close()

        available_points = total_earned - total_redeemed

        reward_catalog = [
            ("TNG Reload Pin RM8", 80),
            ("AEON Voucher RM10", 100),
            ("Shopee Voucher RM10", 100),
            ("GrabFood Voucher RM10", 100),
            ("Lazada Voucher RM10", 100)
        ]

        next_reward_name = None
        next_reward_points = None

        for reward_name, points_required in reward_catalog:
            if available_points < points_required:
                next_reward_name = reward_name
                next_reward_points = points_required
                break

        if next_reward_name is None:
            next_reward_name = "All listed rewards unlocked"
            next_reward_points = available_points if available_points > 0 else 1

        points_needed = max(next_reward_points - available_points, 0)
        progress_value = min(available_points / next_reward_points, 1.0) if next_reward_points > 0 else 0

        if user_info:
            name, email = user_info

            st.markdown(f"## 👋 Hello, {name}")
            st.write(f"**Email:** {email}")

            st.markdown("### 🎯 Reward Progress")

            if next_reward_name == "All listed rewards unlocked":
                st.success("You have enough points to redeem all currently listed rewards.")
                st.progress(1.0)
            else:
                st.info(f"Next Reward: {next_reward_name}")
                st.write(f"You need **{points_needed} more points** to redeem this reward.")
                st.progress(progress_value)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Earned Points", total_earned)
                st.metric("Total Reward Records", total_reward_records)

            with col2:
                st.metric("Available Points", available_points)
                st.metric("Pickup Requests", total_pickup_requests)

            with col3:
                st.metric("Completed Pickups", completed_pickups)
                st.metric("Total Redemptions", total_redemptions)

        else:
            st.error("User info not found.")

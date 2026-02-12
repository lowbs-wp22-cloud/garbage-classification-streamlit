from flask import Flask, request, redirect, url_for, session, render_template_string
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "fyp_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model("FYP_general_waste.h5")
categories = ["paper", "plastic", "metal", "glass", "cardboard", "trash"]

# ---------------- ROUTES ----------------

# LOGIN PAGE
@app.route('/', methods=["GET", "POST"])
def login():
    login_html = '''
    <h2>Login</h2>
    <form method="POST">
        <label>Select Role:</label><br><br>
        <select name="role">
            <option value="USER">USER</option>
            <option value="ADMIN">ADMIN</option>
        </select>
        <br><br>
        <button type="submit">Enter</button>
    </form>
    <p>Don't have account? <a href="/signup">Sign Up</a></p>
    '''
    if request.method == "POST":
        role = request.form["role"]
        session["role"] = role
        return redirect(url_for("dashboard"))
    return render_template_string(login_html)

# SIGNUP PAGE
@app.route('/signup', methods=["GET", "POST"])
def signup():
    signup_html = '''
    <h2>Sign Up</h2>
    <form method="POST">
        <label>Username:</label><br>
        <input type="text" name="username" required><br><br>
        <label>Password:</label><br>
        <input type="password" name="password" required><br><br>
        <label>Select Role:</label><br>
        <select name="role">
            <option value="USER">USER</option>
            <option value="ADMIN">ADMIN</option>
        </select>
        <br><br>
        <button type="submit">Sign Up</button>
    </form>
    <p>Already have account? <a href="/">Login</a></p>
    '''
    if request.method == "POST":
        # In real project, save user info to DB (skipped for simplicity)
        role = request.form["role"]
        session["role"] = role
        return redirect(url_for("dashboard"))
    return render_template_string(signup_html)

# DASHBOARD PAGE
@app.route('/dashboard')
def dashboard():
    role = session.get("role", "USER")
    dashboard_html = f'''
    <h2>Welcome {role}</h2>
    <a href="/category">Choose Garbage Category</a>
    '''
    return render_template_string(dashboard_html)

# CATEGORY SELECTION PAGE
@app.route('/category')
def category():
    category_html = '''
    <h2>Select Garbage Category</h2>
    <a href="/upload/GENERAL_WASTE"><button>GENERAL WASTE</button></a>
    <a href="/upload/FURNITURE"><button>FURNITURE</button></a>
    '''
    return render_template_string(category_html)

# UPLOAD PAGE
@app.route('/upload/<garbage_type>', methods=["GET", "POST"])
def upload(garbage_type):
    upload_html = f'''
    <h2>Upload Garbage Image - {garbage_type}</h2>
    {% if garbage_type == "GENERAL_WASTE" %}
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required><br><br>
        <button type="submit">Predict</button>
    </form>
    {% else %}
    <p>Furniture classification model not available yet.</p>
    {% endif %}
    '''
    if request.method == "POST" and garbage_type == "GENERAL_WASTE":
        file = request.files["image"]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        return render_template_string(f'''
        <h2>Prediction Result</h2>
        <img src="/{filepath}" width="300"><br><br>
        <h3>Predicted Category: {predicted_class}</h3>
        <a href="/category">Back</a>
        ''')
    return render_template_string(upload_html, garbage_type=garbage_type)

# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True)

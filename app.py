from flask import Flask, render_template, request, redirect, session, send_from_directory
import sqlite3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# Load issue detection CNN
model = load_model("model/home_issue_model.h5")

# Image validation model (pretrained on ImageNet)
validator_model = MobileNetV2(weights="imagenet")

HOME_ISSUE_KEYWORDS = [
    "pipe", "faucet", "sink", "toilet",
    "wall", "brick", "concrete", "ceiling",
    "electrical", "switch", "socket", "wire",
    "roof", "floor", "window", "door"
]

app = Flask(__name__)
app.secret_key = "secretkey123"   # required for session

# Ensure uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# ---------- DATABASE CONNECTION ----------
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# ---------- CREATE TABLE (RUNS ONCE) ----------
def create_table():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

create_table()

def is_home_issue_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = validator_model.predict(img)
    decoded = decode_predictions(preds, top=5)[0]

    # Only reject CLEARLY wrong images
    REJECT_KEYWORDS = [
        "person", "man", "woman", "boy", "girl",
        "face", "portrait", "selfie",
        "dog", "cat", "animal"
    ]

    for _, label, confidence in decoded:
        label = label.lower()
        for bad in REJECT_KEYWORDS:
            if bad in label and confidence > 0.50:
                return False

    # Otherwise ACCEPT image for issue analysis
    return True

def detect_issue_type(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Simple color-based clues
    blue_pixels = np.sum(
        (img_np[:, :, 2] > 150) & 
        (img_np[:, :, 1] < 120)
    )

    gray_pixels = np.sum(
        (np.abs(img_np[:, :, 0] - img_np[:, :, 1]) < 10) &
        (np.abs(img_np[:, :, 1] - img_np[:, :, 2]) < 10)
    )

    # Heuristic rules
    if blue_pixels > 500:
        return "Plumbing Leak"

    if gray_pixels > 1000:
        return "Electrical Fault"

    return "Unknown Issue"




# ---------- ROUTES ----------

# Login Page
@app.route('/')
def login():
    return render_template('login.html')

# Register Page
@app.route('/register')
def register():
    return render_template('register.html')

# Register User
@app.route('/register_user', methods=['POST'])
def register_user():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, password)
        )
        conn.commit()
    except:
        return "Username already exists!"
    finally:
        conn.close()

    return redirect('/')

# Login User
@app.route('/login_user', methods=['POST'])
def login_user():
    username = request.form['username']
    password = request.form['password']

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    ).fetchone()
    conn.close()

    if user:
        session['user'] = username
        return redirect('/dashboard')
    else:
        return "Invalid login credentials!"

# Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    else:
        return redirect('/')

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

# Upload Page
@app.route('/upload')
def upload():
    if 'user' in session:
        return render_template('upload.html')
    else:
        return redirect('/')

# Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Handle Image Upload & Show Result
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user' not in session:
        return redirect('/')

    # Get uploaded image
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    # ---------- IMAGE VALIDATION ----------
    if not is_home_issue_image(image_path):
        return render_template(
            "result.html",
            issue="Invalid Image",
            severity="Error",
            message="Please upload a valid home-issue image (plumbing, electrical, or building-related).",
            image_path=None
        )

    # ---------- SIMPLE ISSUE OVERRIDE ----------
    issue = None
    filename_lower = image.filename.lower()

    if "pipe" in filename_lower or "leak" in filename_lower or "water" in filename_lower:
        issue = "Plumbing Leak"

    # ---------- IMAGE PREPROCESSING ----------
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)

    # ---------- MODEL PREDICTION ----------
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # ---------- SEVERITY ----------
    if predicted_class == 1:
        severity = "Minor"
    else:
        severity = "Major"

    # ---------- ISSUE SELECTION ----------
    if issue is None:
        if severity == "Minor":
            issue = "Plumbing Leak"
        else:
            issue = "Electrical Fault"

    # ---------- MESSAGE ----------
    if severity == "Minor":
        message = "This issue can be fixed by yourself. Follow basic repair steps."
    else:
        message = "This issue is complex and requires professional assistance."

    return render_template(
        'result.html',
        issue=issue,
        severity=severity,
        message=message,
        image_path=f"/uploads/{image.filename}"
    )


@app.route('/diy')
def diy():
    if 'user' not in session:
        return redirect('/')

    # Dummy steps (for now)
    issue = "Plumbing Leak"
    steps = [
        "Turn off the main water supply.",
        "Identify the leaking pipe or joint.",
        "Tighten loose connections using a wrench.",
        "Apply plumber tape to seal small leaks.",
        "Turn the water supply back on and check for leaks."
    ]

    return render_template('diy.html', issue=issue, steps=steps)

@app.route('/professional')
def professional():
    if 'user' not in session:
        return redirect('/')

    issue = "Electrical Fault"

    providers = [
        {"name": "Ravi Electrical Services", "type": "Electrician", "location": "Nearby Area"},
        {"name": "SafeHome Repairs", "type": "Electrical Maintenance", "location": "Within 5 km"}
    ]

    return render_template(
        'professional.html',
        issue=issue,
        providers=providers
    )


# ---------- RUN APP ----------
if __name__ == "__main__":
    app.run()


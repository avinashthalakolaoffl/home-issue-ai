from flask import Flask, render_template, request, redirect, session, send_from_directory
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained CNN model
model = load_model("model/home_issue_model.h5")

# IMPORTANT: Must match class_indices from training
CLASS_NAMES = ['electrical', 'plumbing', 'wall_crack']

ISSUE_MAP = {
    'electrical': 'Electrical Fault',
    'plumbing': 'Plumbing Leak',
    'wall_crack': 'Wall Crack'
}

app = Flask(__name__)
app.secret_key = "secretkey123"

# Ensure uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# ---------- DATABASE CONNECTION ----------
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# ---------- CREATE TABLE ----------
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

# ---------- ROUTES ----------

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

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

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    else:
        return redirect('/')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/upload')
def upload():
    if 'user' in session:
        return render_template('upload.html')
    else:
        return redirect('/')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# ---------- IMAGE ANALYSIS ----------
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user' not in session:
        return redirect('/')

    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    # Image preprocessing
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)

    # Model prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    predicted_label = CLASS_NAMES[predicted_class]
    issue = ISSUE_MAP[predicted_label]

    # Severity logic
    if predicted_label == 'plumbing':
        severity = "Minor"
        message = "This issue can be fixed by yourself."
    else:
        severity = "Major"
        message = "This issue requires professional assistance."

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

    issue = "Plumbing Leak"
    steps = [
        "Turn off the main water supply.",
        "Identify the leaking pipe or joint.",
        "Tighten loose connections.",
        "Apply plumber tape if necessary.",
        "Turn water supply back on and test."
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

if __name__ == "__main__":
    app.run()

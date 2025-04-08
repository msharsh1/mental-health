# Mental Health Tracker Web App

from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
import json
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Helper functions for JSON storage
def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_data('users.json')
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    users = load_data('users.json')
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return 'Username already exists'
        users[username] = password
        save_data('users.json', users)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user_moods = load_data('moods.json')

    if request.method == 'POST':
        mood = request.form['mood']
        entry = {'mood': mood, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        user_moods.setdefault(username, []).append(entry)
        save_data('moods.json', user_moods)

    moods = user_moods.get(username, [])
    return render_template('dashboard.html', moods=moods, username=username)


import pickle
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            with open('mental_health_model.pkl', 'rb') as f:
                model_dep, model_anx, feature_cols = pickle.load(f)

            # Get all form inputs
            fields = [
                'sleep','appetite','focus','fatigue','mood_swings','social_interaction','stress',
                'irritability','physical_symptoms','self_esteem','crying_spells','suicidal_thoughts',
                'motivation','daily_functioning','panic_attacks']

            input_dict = {col: 0 for col in feature_cols}

            for field in fields:
                value = request.form.get(field)
                if value:
                    key = f"{field}_{value}"
                    if key in input_dict:
                        input_dict[key] = 1

            input_df = pd.DataFrame([input_dict])

            # Make predictions
            pred_dep = model_dep.predict(input_df)[0]
            pred_anx = model_anx.predict(input_df)[0]

            result = []
            if pred_dep:
                result.append("Depression")
            if pred_anx:
                result.append("Anxiety")
            if not result:
                result.append("No major symptoms detected.")

            return render_template('predict.html', result=", ".join(result))

        except Exception as e:
            return f"Error during prediction: {e}"

    return render_template('predict.html')


from flask import send_from_directory
@app.route('/findDoctors')
def serve_find_doctors():
    return send_from_directory('findDoctors', 'index.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

# Let me know if you want to add features like charts!
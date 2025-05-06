import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = '8106479937'  # Change this to a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Using SQLite

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to login if unauthorized

# Load the trained ML models
sc_model = joblib.load('notebooks/scaler.pkl')  
random_forest_model = joblib.load('notebooks/random_forest.pkl')  

# User model for authentication
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict_page'))
        else:
            flash('Invalid username or password!', 'danger')
    return render_template('login.html')

# Register route with unique username check
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        
        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already taken. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        
        # Hash password
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        
        # Create new user
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Prediction page (protected)
@app.route('/predict_page')
@login_required
def predict_page():
    return render_template('index.html')

# Prediction logic
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Extract form values and convert them to float
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        final_scaled = sc_model.transform(final_features)
        prediction = random_forest_model.predict(final_scaled)

        # Interpret prediction result
        answer = "ML suggests potential autism traits." if prediction == 1 else "ML analysis does not indicate autism traits."
        return render_template('index.html', prediction_text=answer)

    except Exception as e:
        # If error occurs, return the error message
        return str(e)

# Initialize database
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)


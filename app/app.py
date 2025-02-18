from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Import or define the custom LinearRegression class before loading the model
from model_file import LinearRegression  # Replace 'your_model_file' with the actual file where the class is defined
from model_file import NoPenalty

app = Flask(__name__)

# Load models and scaler
old_model = joblib.load("random_forest_model.pkl")
new_model = joblib.load("A2_car.pkl") 
scaler = joblib.load("scaler.pkl")

FEATURES = ['km_driven', 'owner', 'mileage', 'engine', 'max_power']
features_to_scale = ['km_driven', 'mileage', 'engine', 'max_power']  # Only these are scaled

owner_mapping = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/old_model', methods=['GET', 'POST'])
def old_model_page():
    prediction = None
    if request.method == 'POST':
        # Get user input
        input_data = {feature: float(request.form[feature]) for feature in features_to_scale}
        owner_input = owner_mapping[request.form['owner']]  # Convert dropdown value to integer

        # Scale only the necessary features
        scaled_values = scaler.transform([[input_data[f] for f in features_to_scale]])

        # Combine scaled and unscaled features
        input_scaled = np.hstack([scaled_values, [[owner_input]]])  # Owner is not scaled

        # Predict and apply inverse log transformation
        log_price = old_model.predict(input_scaled)[0]
        prediction = np.exp(log_price)

    return render_template('old_model.html', prediction=prediction, features=FEATURES)

@app.route('/new_model', methods=['GET', 'POST'])
def new_model_page():
    prediction = None
    message = "Our new model has been trained with optimized parameters and improved data preprocessing, leading to more accurate price predictions compared to the old model."
    
    if request.method == 'POST':
        # Get user input
        input_data = {feature: float(request.form[feature]) for feature in features_to_scale}
        owner_input = owner_mapping[request.form['owner']]  # Convert dropdown value to integer

        # Scale only the necessary features
        scaled_values = scaler.transform([[input_data[f] for f in features_to_scale]])

        # Combine scaled and unscaled features
        input_scaled = np.hstack([scaled_values, [[owner_input]]])  # Owner is not scaled

        # Predict and apply inverse log transformation
        log_price = new_model.predict(input_scaled)[0]
        prediction = np.exp(log_price)

    return render_template('new_model.html', prediction=prediction, message=message, features=FEATURES)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=788)

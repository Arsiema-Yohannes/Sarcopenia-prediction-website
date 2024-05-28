from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the models
models = {
    "Male": {
        "1": joblib.load('models/ensemble_model1M.pkl'),
        "2": joblib.load('models/ensemble_model2M.pkl'),
        "3": joblib.load('models/ensemble_model3M.pkl')
    },
    "Female": {
        "1": joblib.load('models/ensemble_model1F.pkl'),
        "2": joblib.load('models/ensemble_model2F.pkl'),
        "3": joblib.load('models/ensemble_model3F.pkl')
    }
}

# Render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form.get('gender')
    model = request.form.get('model')
    
    # Get the selected model
    selected_model = models[gender][model]
    
    # Extract input features
    features = [
        'MMSE', 'Age', 'Weight', 'Height', 'Waist', 'Hip', 'Smoking',
        'Smoking (packet/year)', 'Alcohol', 'DM', 'DM duration', 'İnsülin', 'Hiperlipidemi', 'Dyslipidemia duration','KAH', 'KAH duration', 'Hipotiroidi', 'ASTIM',
        'HT', 'HT duration','Education', 'Occupation', 'Working Status', 'Exercise'
    ]
    
    
    if model == '2':
        features.extend(['CST', 'LowCST', 'Gait speed'])
    elif model == '3':
        features.extend(['CST', 'LowCST', 'Gait speed', 'Low grip strength', 'Grip strength'])
    
    # Retrieve input values
    input_values = [request.form.get(feature) for feature in features]
    
    # Convert input values to float
    input_values = [float(value) if value else 0.0 for value in input_values]
    
    # Predict probability
    probability = selected_model.predict_proba([input_values])[0, 1]
    
    # Determine prediction
    if probability >= 0.8:
        prediction = 'Positive'
    elif probability <= 0.2:
        prediction = 'Negative'
    else:
        prediction = 'Further testing required'
    
    return render_template('result.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)

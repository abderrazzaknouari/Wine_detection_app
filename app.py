from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)

# Load the trained random forest model
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("StandardScaler.pkl")

#Define the min and max values for the input features
featuresLabel = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
            'pH', 'sulphates', 'alcohol']
defaultValues = [7.4, 0.25, 0.29, 2.2, 0.05406,	19,	49,	0.99666, 3.4, 0.76,	10.9]
min_values = [4, 0.1, 0, 0.5, 0, 1, 5, 0, 2, 0, 5]  
max_values = [20, 2, 1, 20, 1, 100, 300, 2, 5, 2, 20]
steps = [0.1, 0.001, 0.01, 0.1, 0.001, 1, 1, 0.00001, 0.01, 0.01, 0.1]

@app.route('/')
def home():
    return render_template('index.html', min_values=min_values, 
                           max_values=max_values, steps=steps, 
                           featuresLabel=featuresLabel, defaultValues=defaultValues)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]  # Extract features from the form
    defaultValues = features #change default values to the ones entered by the user
    features = np.array(features).reshape(1, -1)  # Reshape features for prediction
    features_scaled = scaler.transform(features)  # Reshape features for prediction
    prediction = model.predict(features)[0]  # Make prediction
    if prediction == 0:
        result = 'Bad'
    else:
        result = 'Good'
    features_scaled
    return render_template('index.html', prediction_text=f'Predicted wine quality: {result}', 
                           min_values=min_values, max_values=max_values, 
                           steps=steps, featuresLabel=featuresLabel, defaultValues=defaultValues)

    
if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from utils import region_mapping, age_mapping, property_type_mapping, status_mapping

app = Flask(__name__)

# Load the trained model
model = joblib.load("Price_prediction.pkl")

# Extract region names for frontend dropdown
regions = sorted(region_mapping.keys())  # Get sorted region list

@app.route('/')
def home():
    return render_template('index.html', regions=regions)  # Pass regions to template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend
        data = request.json
        bhk = int(data['bhk'])
        area = float(data['area'])
        property_type = property_type_mapping[data['property_type']]
        region = region_mapping[data['region']]
        status = status_mapping[data['status']]
        age = age_mapping[data['age']]

        # Prepare input data
        input_data = pd.DataFrame([[bhk, property_type, area, region, status, age]],
                                  columns=['bhk', 'type', 'area', 'region', 'status', 'age'])

        # Predict price
        predicted_price = model.predict(input_data)

        return jsonify({'predicted_price': round(predicted_price[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


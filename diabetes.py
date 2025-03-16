from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
diabetes = Flask(__name__)

# Load the model and scaler from the pickle file
try:
    with open('diabetes_prediction_model.pkl', 'rb') as file:
        model, scaler = pickle.load(file)  # Ensure model and scaler are saved together
except FileNotFoundError:
    print("Error: Model file 'diabetes_prediction_model.pkl' not found.")
    model = None
    scaler = None

# Route: Home Page with Form
@diabetes.route('/')
def home():
    return render_template('diabetes.html')

# Route: Prediction
@diabetes.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded properly."}), 500

    try:
        # Extract input values from the form
        pregnancy = int(request.form['pregnancy'])
        glucose = float(request.form['glucose'])
        bp = float(request.form['bp'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = int(request.form['age'])

        # Convert input values to a NumPy array
        input_data = np.array([[pregnancy, glucose, bp, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

        # Scale the input data using the pre-fitted scaler
        scaled_data = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_data)[0]
        output = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Render the result on the HTML page
        return render_template('diabetes.html', prediction_text=output)

    except ValueError as ve:
        # Handle errors related to type conversion
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400

    except Exception as e:
        # Handle general exceptions
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    diabetes.run(debug=True, port=5009)

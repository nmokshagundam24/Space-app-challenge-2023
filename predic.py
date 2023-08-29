# Python (Flask server)
from flask import Flask, request, jsonify, render_template
import Wildfires  # Your module containing ML model and prediction logic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from the POST request
    prediction = your_ml_module.predict(data)  # Use your prediction logic
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

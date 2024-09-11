import sys
import os
from flask import Flask, request, jsonify
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.predict import predict


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files or 'brand' not in request.form:
        return jsonify({"error": "Missing image or brand"}), 400

    image = request.files['image']
    brand = request.form['brand']

    # Вызов predict
    try:
        result = predict(image, brand)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

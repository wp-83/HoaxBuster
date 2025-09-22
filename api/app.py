import os
import sys
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# import preprocessor
from model.preprocessing import IndonesianTextPreprocessor

# load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multinomialnb.joblib")
model = joblib.load(MODEL_PATH)

# make instance of class preprocessor
preprocessor = IndonesianTextPreprocessor()

# initialize the Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# endpoint default
@app.route("/")
def home():
    return {"message": "Hoax Detector API (Flask)"}

# endpoint prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "information" not in data:
            return jsonify({"error": "Field 'information' must exist"}), 400

        information = data["information"]
        clean_text = preprocessor._preprocess_text(information)
        prediction = model.predict_proba([clean_text])[0]

        return jsonify({
            "input": information,
            "processed": clean_text,
            "prediction": str("{:,.4f}".format(prediction[1])) #take the hoax probability
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# FOR LOCAL HOST RUNNING
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
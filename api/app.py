import os
import sys
from flask import Flask, request, jsonify
import joblib

# add root path of this project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# import preprocessor
from model.preprocessing import IndonesianTextPreprocessor

# path model
MODEL_PATH = os.path.join(ROOT_DIR, "model", "final_model", "multinomialnb.joblib")

# check if the model is found or not
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model is not found: {MODEL_PATH}")

# load model
model = joblib.load(MODEL_PATH)

# make instance of class preprocessor
preprocessor = IndonesianTextPreprocessor()

# initialize the Flask
app = Flask(__name__)

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
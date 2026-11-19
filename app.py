from flask import Flask, request, jsonify
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load trained model
model = joblib.load("model/iris_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris ML Model API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        features = data["features"]

        prediction = model.predict([features])

        logging.info(
            f"Prediction request received: {features}"
        )

        logging.info(
            f"Prediction generated: {int(prediction[0])}"
        )

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:

        logging.error(
            f"Prediction error: {str(e)}"
        )

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
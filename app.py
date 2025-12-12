"""
Flask web application for Garcon Hotel Cancellation Prediction Model.
"""
import os
from flask import Flask, render_template, request, jsonify

from garcon_model.predict import make_prediction

app = Flask(__name__)


@app.route("/")
def index():
    """Render the main prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests from the form."""
    try:
        # Extract form data
        input_data = {
            "no_of_adults": [int(request.form.get("no_of_adults", 1))],
            "no_of_children": [float(request.form.get("no_of_children", 0))],
            "no_of_weekend_nights": [float(request.form.get("no_of_weekend_nights", 0))],
            "no_of_week_nights": [float(request.form.get("no_of_week_nights", 1))],
            "type_of_meal_plan": [request.form.get("type_of_meal_plan", "Not Selected")],
            "required_car_parking_space": [float(request.form.get("required_car_parking_space", 0))],
            "room_type_reserved": [request.form.get("room_type_reserved", "Room_Type 1")],
            "lead_time": [float(request.form.get("lead_time", 0))],
            "arrival_month": [int(request.form.get("arrival_month", 1))],
            "arrival_date": [int(request.form.get("arrival_date", 1))],
            "arrival_year": [int(request.form.get("arrival_year", 2024))],
            "market_segment_type": [request.form.get("market_segment_type", "Online")],
            "repeated_guest": [int(request.form.get("repeated_guest", 0))],
            "no_of_previous_cancellations": [int(request.form.get("no_of_previous_cancellations", 0))],
            "avg_price_per_room": [float(request.form.get("avg_price_per_room", 100))],
            "no_of_special_requests": [int(request.form.get("no_of_special_requests", 0))],
        }

        # Make prediction
        result = make_prediction(input_data=input_data)

        if result["errors"]:
            return render_template(
                "index.html",
                error=str(result["errors"]),
                form_data=request.form,
            )

        # Interpret prediction result
        prediction_value = result["predictions"][0]
        # The model returns exp(prediction), where 0 = not canceled, 1 = canceled
        # Values close to 1 indicate not canceled, values close to e (~2.718) indicate canceled
        is_canceled = prediction_value > 1.5
        probability = min((prediction_value - 1) / 1.718 * 100, 100) if prediction_value > 1 else 0

        return render_template(
            "index.html",
            prediction=is_canceled,
            probability=round(probability, 1),
            version=result["version"],
            form_data=request.form,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            form_data=request.form,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for predictions."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = make_prediction(input_data=data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint for Heroku."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

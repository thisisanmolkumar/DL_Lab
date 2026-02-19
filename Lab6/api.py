from flask import Flask, render_template, request, jsonify
from predict import predict

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", result=None, error=None)


@app.post("/predict")
def predict_route():
    text = request.form.get("sentence", "").strip()
    if not text:
        return render_template("index.html", result=None, error="Please enter a sentence.")

    try:
        result = predict(text)
        return render_template("index.html", result=result, error=None)
    except Exception as e:
        return render_template("index.html", result=None, error=f"Error: {e}")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("sentence") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'sentence'"}), 400

    try:
        return jsonify(predict(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # http://127.0.0.1:5000
    app.run(debug=True)

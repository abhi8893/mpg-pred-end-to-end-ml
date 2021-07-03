from flask import Flask, request, jsonify
from src.project.predict import predict_mpg


app = Flask('app')

@app.route('/')
def index():
    return 'MPG Prediction ML Model Homepage'

@app.route('/ping')
def ping():
    return 'Pinging ML Model!'

@app.route('/predict', methods=['POST'])
def predict():
    vehicle_config = request.get_json()
    preds = predict_mpg(vehicle_config)
    result = {'mpg_prediction': list(preds)}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

    





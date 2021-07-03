import requests

PORT = 9696
URL = f'http://localhost:{PORT}/predict'

vehicle_config = {
    'acceleration': [17.0],
    'cylinder': [6],
    'displacement': [250.0],
    'horsepower': [100.0],
    'model_year': [74],
    'origin': [1],
    'weight': [3336.0]
}

resp = requests.post(URL, json=vehicle_config)
print(resp.text)
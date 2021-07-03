import requests
import sys

server_type = sys.argv[1]

PORT = 9696

if server_type == 'local':
    URL = f'http://localhost:{PORT}/predict'
elif server_type == 'heroku':
    URL = f'https://mpg-pred-ml.herokuapp.com/predict'

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
import requests

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('path to image.jpg', 'rb')}

response = requests.post(url, files=files)
print(response.json())

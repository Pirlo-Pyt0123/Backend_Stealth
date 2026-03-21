import requests

url = "http://localhost:8000/detect"
with open("output/personUnreal2.png", "rb") as f:
    files = {"file": ("frame_001.jpg", f, "image/jpeg")}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.json())

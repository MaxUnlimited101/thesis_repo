import requests

def request_one_pred():
    url = "http://localhost:8000/predict"
    files = {"file": open("photo.jpg", "rb")}
    response = requests.post(url, files=files)
    print(response.json())
    

def request_multiple_pred():
    url = "http://localhost:8000/predict_batch"
    files = [
        ("files", open("photo.jpg", "rb")),
        ("files", open("photo.jpg", "rb"))
    ]
    response = requests.post(url, files=files)
    print(response.json())
    

if __name__ == "__main__":
    request_one_pred()
    request_multiple_pred()
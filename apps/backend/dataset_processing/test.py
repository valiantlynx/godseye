import requests

url = "http://127.0.0.1:8000/process_video"
video_path = "./sample/1.mp4"

# Open the video file in binary mode
with open(video_path, "rb") as f:
    files = {
        "source": (video_path, f, "video/mp4")
    }
    data = {
        "model_path": "./models/yolov8n-pose.pt",
        "conf": "0.3"
    }
    response = requests.post(url, files=files, data=data)

# Print response
print(response.json())

# bro idk

import os
import io
import cv2
import uvicorn
from PIL import Image
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, WebSocket
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from dotenv import load_dotenv
from utils.gmail import send_gmail
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_training.test import ViolenceDetector


load_dotenv()


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])

# Load YOLO model
model = YOLO("models/yolov8n-pose.pt")
# loaf the violence model
detector = ViolenceDetector("models/acc_96__loss_0.1__Epochs_30.h5")
# number of frames before detection
num_frames = 20
# the frames to json
json_data = []
# when creating the json we need labels
labeledKeypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def frame_to_json(frame, frame_index=0):
    # Detect keypoints using YOLO model
    results = model(frame)
    newFrameData = []

    # Extract keypoints and bounding box data for JSON output
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box_data = boxes.xyxy[i].cpu().numpy()
                confidence = boxes.conf[i].cpu().item()
                box = {
                    "x1": float(box_data[0]),
                    "y1": float(box_data[1]),
                    "x2": float(box_data[2]),
                    "y2": float(box_data[3])
                }

                keypoints_data = []
                if keypoints is not None:
                    keypoints_array = keypoints.data[i].cpu().numpy()
                    for j, (x, y, conf) in enumerate(keypoints_array):
                        keypoints_data.append({
                            "label": labeledKeypoints[j],
                            "coordinates": {"x": float(x), "y": float(y)},
                            "confidence": float(conf)
                        })

                newFrameData.append({
                    "person_id": i + 1,
                    "confidence": confidence,
                    "box": box,
                    "keypoints": keypoints_data
                })

    return ({"frame": frame_index, "detections": newFrameData})


@app.websocket("/ws/video-stream/")
async def video_stream(websocket: WebSocket):
    global json_data
    await websocket.accept()
    try:
        while True:
            # Receive frame data from client
            frame_data = await websocket.receive_bytes()

            # Decode the frame from bytes to image
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                break

            # convert the frames to json data
            json_data.append(frame_to_json(frame, len(json_data)))

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(json_data) < num_frames:
                await websocket.send_json({"job": "processing"})
                continue

            # run the violence prediction model on the json data
            result = detector.predict(json_data)
            json_data.clear()

            # print results
            if "error" not in result:
                print(f"Violence Probability: {result['probability']:.2%}")
                print(f"Classification: {'Violent' if result['is_violent'] else 'Non-violent'}")
                print(f"Confidence: {result['confidence']:.2%}")
                await websocket.send_json({"result": f"{'Violent' if result['is_violent'] else 'Non-violent'}", "confidence": f"{result["confidence"]:.2%}"})
            else:
                print(f"Error processing: {result['error']}")
                await websocket.send_json({"result": f"{result['error']}"})

    except Exception as e:
        print("Connection closed:", e)
    finally:
        await websocket.close()


@app.get("/")
def read_root():
    return {"Welcome": "FastAPI API for Pose object detection, add '/docs' for endpoints."}


@app.post("/process_image")
async def process_image(
    source: UploadFile = File(...),
    conf: float = 0.3
):
    # Check the file extension
    file_extension = os.path.splitext(source.filename)[1]

    # Process the file
    results = []
    if file_extension.lower() not in (".jpg", ".jpeg", ".png"):
        return {"error": "Unsupported file format"}

    # read the incomming bytes
    source_bytes = await source.read()

    # convert and decode
    image_array = np.frombuffer(source_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Failed to process the image"}

    json_data = frame_to_json(image)
    violence = detector.predict(json_data)

    # print results
    if "error" not in result:
        print(f"Violence Probability: {result['probability']:.2%}")
        print(f"Classification: {'Violent' if result['is_violent'] else 'Non-violent'}")
        print(f"Confidence: {result['confidence']:.2%}")
        return {
            "probability": result['probability'],
            "classification": "Violent" if result['is_violent'] else "Non-violent",
            "confidence": result['confidence']
        }

    else:
        print(f"Error processing: {result['error']}")
        return {"error": result['error']}


@app.post("/process_video")
async def process_video(
    source: UploadFile = File(...),
    conf: float = 0.3
):
    # Check the file extension
    file_extension = os.path.splitext(source.filename)[1]
    if file_extension.lower() not in (".mp4", ".avi", ".mov", ".webm"):
        return {"error": "Unsupported file format"}

    # Save the video temporarily to process it
    temp_video_path = f"temp_{source.filename}"
    with open(temp_video_path, "wb") as video_file:
        video_file.write(await source.read())

    # Open the video with OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    json_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        json_data.append(frame_to_json(frame))

    cap.release()
    os.remove(temp_video_path)  # Clean up the temporary video file

    if len(json_data) == 0:
        return {"error": "Failed to process the video"}

    # Run the violence prediction model on the JSON data
    result = detector.predict(json_data)

    # Check if prediction was successful and format the response
    if "error" not in result:
        print(f"Violence Probability: {result['probability']:.2%}")
        print(f"Classification: {'Violent' if result['is_violent'] else 'Non-violent'}")
        print(f"Confidence: {result['confidence']:.2%}")
        return {
            "probability": result['probability'],
            "classification": "Violent" if result['is_violent'] else "Non-violent",
            "confidence": result['confidence']
        }
    else:
        print(f"Error processing: {result['error']}")
        return {"error": result['error']}


@app.post("/send_email")
async def send_notification(recipient_email: str, message: str):
    """ send mails to recipients """
    if not recipient_email:
        return {"error": "Missing recipient"}, 400
    result = send_gmail(recipient_email, message)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

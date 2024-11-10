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


load_dotenv()


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])

# Load YOLO model
model = YOLO("../dataset_processing/models/yolov8n-pose.pt")


def generate_frames(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.websocket("/ws/video-stream/")
async def video_stream(websocket: WebSocket):
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

            results = model(frame)
            keypoints_tensor = results[0].keypoints.xy  # Extract xy keypoints

            keypoints_data = [
                {"x": float(point[0]), "y": float(point[1])}
                for point in keypoints_tensor[0]  # Iterate over each keypoint in the frame
                if point[0] != 0 and point[1] != 0  # Filter out zero points
            ]

            await websocket.send_json({"keypoints": keypoints_data})

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

    results = model(source=image, conf=conf)

    # violence = lstm(resutls)
    # return violence

    # this will be swapped out for the actual lstm model result:    
    processed_images = []
    for idx, r in enumerate(results):
        image_array = r.plot(conf=True, boxes=True)

        # Encode the image in memory as PNG
        success, buffer = cv2.imencode('.png', image_array)
        if success:
            # Create an in-memory bytes buffer
            image_bytes = io.BytesIO(buffer.tobytes())
            processed_images.append(image_bytes)

    image_bytes.seek(0)
    return StreamingResponse(image_bytes, media_type="image/png")

@app.post("/process_video")
async def process_video(
    source: UploadFile = File(...),
    conf: float = 0.3
):
    # Save the uploaded video
    file_extension = os.path.splitext(source.filename)[1]
    temp_file_path = "temp_video" + file_extension
    with open(temp_file_path, "wb") as f:
        f.write(source.file.read())

    # Process the video
    if file_extension.lower() in (".mp4", ".avi", ".mov", ".mkv", "webm"):
        results = model(source=temp_file_path, show=True, conf=conf)
    else:
        return {"error": "Unsupported file format"}

    # Save processed video frames
    processed_frames = []
    for idx, r in enumerate(results):
        image_array = r.plot(conf=True, boxes=True)
        processed_frames.append(image_array)

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    processed_video_path = "./sample/processed_video.avi"

    out = cv2.VideoWriter(processed_video_path, fourcc, 30,
                          (processed_frames[0].shape[1], processed_frames[0].shape[0]))

    # Write processed frames to the video
    for frame in processed_frames:
        out.write(frame)
    out.release()
    # Read the processed video as bytes
    with open(processed_video_path, "rb") as video_file:
        video_bytes = video_file.read()

    # Adjust media_type as needed
    # return processed_video_path(processed_video_path=processed_video_path)
    # for future read and stream in api or save in s3 bucket and fetch
    # # Read the processed video as bytes
    # with open(processed_video_path, "rb") as video_file:
    #     video_bytes = video_file.read()

    # return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4")
    return {"result": f"Succesfully saved in '{str(processed_video_path)}'"}


@app.post("/send_email")
async def send_notification(recipient_email: str, message: str):
    """ send mails to recipients """
    if not recipient_email:
        return {"error": "Missing recipient"}, 400
    result = send_gmail(recipient_email, message)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

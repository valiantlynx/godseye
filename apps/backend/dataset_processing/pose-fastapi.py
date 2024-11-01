import os
import io
import cv2
import uvicorn
from PIL import Image
# from typing import Optional
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])

# Load YOLO model
model = YOLO("./models/yolov8n-pose.pt")

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

@app.get("/stream_video")
async def stream_video():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def read_root():
    return {"Welcome": "FastAPI API for Pose object detection"}


@app.post("/process_image")
async def process_files(
    source: UploadFile = File(...),
    model_path: str = "./models/yolov8n-pose.pt",
    conf: float = 0.3
):
    # Save the uploaded file
    file_extension = os.path.splitext(source.filename)[1]
    temp_file_path = "temp_file" + file_extension
    with open(temp_file_path, "wb") as f:
        f.write(source.file.read())

    # Load the YOLO model
    model = YOLO(model_path)

    # Process the file
    results = []
    if file_extension.lower() in (".jpg", ".jpeg", ".png"):
        results = model(source=temp_file_path, conf=conf)
    else:
        return {"error": "Unsupported file format"}

    processed_images = []
    for idx, r in enumerate(results):
        image_array = r.plot(conf=True, boxes=True)
        # Save the image as a file
        cv2.imwrite("processed_image.png", image_array)
        with open("processed_image.png", "rb") as image_file:
            image_bytes = image_file.read()
            processed_images.append(image_bytes)

    return StreamingResponse(io.BytesIO(processed_images[0]), media_type="image/png")


@app.post("/process_video")
async def process_video(
    source: UploadFile = File(...),
    model_path: str = "./models/yolov8n-pose.pt",
    conf: float = 0.3
):
    # Save the uploaded video
    file_extension = os.path.splitext(source.filename)[1]
    temp_file_path = "temp_video" + file_extension
    with open(temp_file_path, "wb") as f:
        f.write(source.file.read())

    # Load the YOLO model
    model = YOLO(model_path)

    # Process the video
    if file_extension.lower() in (".mp4", ".avi", ".mov", ".mkv"):
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# client side example:

import cv2
import asyncio
import websockets
import numpy as np
import json

# Server WebSocket URL
ws_url = "ws://localhost:8000/ws/video-stream/"


async def stream_video():
    async with websockets.connect(ws_url) as websocket:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Send frame to server
            await websocket.send(frame_bytes)

            # Receive keypoints from server
            keypoints_data = await websocket.recv()
            keypoints = json.loads(keypoints_data)["keypoints"]
            print("Keypoints:", keypoints)

        cap.release()

# Run the video stream
asyncio.run(stream_video())

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import random

# Load MoveNet MultiPose model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1").signatures['serving_default']

# Predefined colors for up to 6 people (can add more if needed)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def detect_keypoints(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # Pass the input image using the named argument "input"
    outputs = model(input=input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.3):
    height, width, _ = frame.shape
    people = keypoints_with_scores[0]  # Shape: [6,56]

    for idx, person in enumerate(people):
        person_score = person[55]
        if person_score < confidence_threshold:
            continue

        # Assign a color for each person based on index
        color = COLORS[idx % len(COLORS)]  # Cycle through colors if more than predefined

        # Extract keypoints: indices 0 to 50 (17 keypoints x 3)
        keypoints = person[:51].reshape((17, 3))
        for kp in keypoints:
            y, x, kp_confidence = kp
            if kp_confidence > confidence_threshold:
                cv2.circle(frame, (int(x * width), int(y * height)), 5, color, -1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    keypoints_with_scores = detect_keypoints(frame)
    draw_keypoints(frame, keypoints_with_scores)
    cv2.imshow('MoveNet Multi-Person Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

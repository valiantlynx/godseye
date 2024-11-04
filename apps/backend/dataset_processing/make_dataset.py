import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load MoveNet MultiPose model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1").signatures['serving_default']

# Define dataset paths
input_root_dir = 'archive/RWF-2000'  # Original dataset directory
output_root_dir = 'archive/processed_RWF-2000_movenet'  # Directory to save processed dataset

# Define categories and subcategories
sets = ['train', 'val']
categories = ['Fight', 'NonFight']

# Predefined colors for up to 6 people (adjust or add more if needed)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def detect_keypoints(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # Pass the input image to the model
    outputs = model(input=input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    
    # Debugging: Print the shape and a sample of the keypoints data
    logging.info(f"Keypoints shape: {keypoints_with_scores.shape}")
    logging.info(f"Sample keypoints data: {keypoints_with_scores[0][:3]}")  # Print first few keypoints for inspection

    return keypoints_with_scores

def draw_keypoints_on_frame(frame, keypoints_with_scores, threshold=0.3):
    for person_id, person in enumerate(keypoints_with_scores[0]):
        color = COLORS[person_id % len(COLORS)]
        
        # Extract only the first 51 values and reshape to (17, 3)
        keypoints = np.reshape(person[:51], (-1, 3))
        
        for x, y, confidence in keypoints:
            if confidence > threshold:
                # Convert normalized x, y coordinates to pixel coordinates
                x_pixel = int(x * frame.shape[1])
                y_pixel = int(y * frame.shape[0])
                
                # Draw the keypoint
                cv2.circle(frame, (x_pixel, y_pixel), 4, color, -1)
                
    return frame

# Process a limited number of videos
LIMIT = 3  # Number of videos to process per category

# Loop through each set and category
for set_name in sets:
    for category in categories:
        input_dir = os.path.join(input_root_dir, set_name, category)
        output_dir = os.path.join(output_root_dir, set_name, category)
        os.makedirs(output_dir, exist_ok=True)

        video_count = 0

        # Loop through videos in the current category
        for video_file in os.listdir(input_dir):
            if video_file.endswith('.avi') and video_count < LIMIT:
                video_name = os.path.splitext(video_file)[0]
                video_input_path = os.path.join(input_dir, video_file)

                # Create a subfolder for each video in the output directory
                video_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)

                # Paths for saving original and processed videos, and keypoints
                original_video_path = os.path.join(video_output_dir, 'original.avi')
                processed_video_path = os.path.join(video_output_dir, 'processed.avi')
                
                # Log the start of processing for the video
                logging.info(f"Processing {set_name}/{category}/{video_file}")

                # Open the video
                cap = cv2.VideoCapture(video_input_path)
                out = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Detect keypoints and draw them on the frame
                    keypoints_with_scores = detect_keypoints(frame)
                    frame_with_keypoints = draw_keypoints_on_frame(frame, keypoints_with_scores)

                    # Initialize video writer if not already set up
                    if out is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(processed_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

                    out.write(frame_with_keypoints)

                cap.release()
                if out:
                    out.release()
                
                # Log completion of the video
                logging.info(f"Completed processing {set_name}/{category}/{video_file}")
                video_count += 1

            if video_count >= LIMIT:
                break  # Stop after processing the limit

logging.info("Processing completed.")

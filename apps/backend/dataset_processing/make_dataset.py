import os
import json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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
    print(f"Keypoints shape: {keypoints_with_scores.shape}")
    print("Sample keypoints data:", keypoints_with_scores[0][:3])  # Print first few keypoints for inspection

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


# Loop through each set and category
for set_name in sets:
    for category in categories:
        input_dir = os.path.join(input_root_dir, set_name, category)
        output_dir = os.path.join(output_root_dir, set_name, category)
        os.makedirs(output_dir, exist_ok=True)

        # Loop through videos in the current category
        for video_file in os.listdir(input_dir):
            if video_file.endswith('.avi'):  # Process only .avi files
                video_name = os.path.splitext(video_file)[0]
                video_input_path = os.path.join(input_dir, video_file)

                # Create a subfolder for each video in the output directory
                video_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)

                # Paths for saving original and processed videos, and keypoints
                original_video_path = os.path.join(video_output_dir, 'original.avi')
                processed_video_path = os.path.join(video_output_dir, 'processed.avi')
                keypoints_path = os.path.join(video_output_dir, 'keypoints.jsonl')

                # Open the input video
                cap = cv2.VideoCapture(video_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Set up video writers for original and processed videos
                out_original = cv2.VideoWriter(original_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                out_processed = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

                # Open keypoints file for writing
                with open(keypoints_path, 'w') as kp_file:
                    frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Save the original frame
                        out_original.write(frame)

                        # Run MoveNet to get keypoints and process the frame
                        keypoints_with_scores = detect_keypoints(frame)

                        # Save keypoints for this frame to JSONL file
                        json.dump({"frame": frame_idx, "keypoints": keypoints_with_scores.tolist()}, kp_file)
                        kp_file.write('\n')

                        # Draw keypoints on frame and save to processed video
                        frame_with_keypoints = draw_keypoints_on_frame(frame, keypoints_with_scores)
                        out_processed.write(frame_with_keypoints)

                        frame_idx += 1

                # Release resources
                cap.release()
                out_original.release()
                out_processed.release()

print(f"Processed dataset saved in '{output_root_dir}' directory.")

import os
import json
import numpy as np
import logging
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Parameters
root_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "dataset_processing", "archive", "processed_RWF-2000")  # Root directory of the dataset
no_of_timesteps = 20  # Number of frames in each sequence
keypoint_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Initialize dataset lists
X = []
y = []

# Helper function to load JSON keypoints data
def load_json_data(json_path, label):
    try:
        with open(json_path) as file:
            data = json.load(file)
            frames_data = []

            # Log the length and structure of the data
            logging.debug(f"Loaded {json_path}: {len(data)} frames")

            # Ensure we have enough frames for the specified timestep
            if len(data) < no_of_timesteps:
                logging.warning(f"Skipping {json_path} as it has fewer than {no_of_timesteps} frames.")
                return None  # Skip sequences shorter than required timesteps

            # Create sequences of keypoints
            for i in range(no_of_timesteps, len(data)):
                sequence = []
                frames = data[i - no_of_timesteps:i]  # Get a sequence of frames

                for frame in frames:
                    for detection in frame["detections"]:
                        person_keypoints = []
                        for label in keypoint_labels:
                            kp = next((kp for kp in detection["keypoints"] if kp["label"] == label), None)
                            if kp:
                                person_keypoints.extend([kp["coordinates"]["x"], kp["coordinates"]["y"]])
                            else:
                                person_keypoints.extend([0, 0])  # Fill missing keypoints with zeros

                        sequence.append(person_keypoints)  # Add person's keypoints for this frame

                frames_data.append(sequence)

            # Return data only if we have sufficient sequences
            if len(frames_data) > 0:
                return frames_data
            else:
                logging.warning(f"No valid sequences found in {json_path}.")
                return None

    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        return None

# Traverse through Fight and NonFight directories
for set_type in ["train", "val"]:
    for category in ["Fight", "NonFight"]:
        category_dir = os.path.join(root_dir, set_type, category)
        label = 1 if category == "Fight" else 0

        # Ensure the directory exists
        if not os.path.exists(category_dir):
            logging.error(f"Directory not found: {category_dir}")
            continue

        logging.info(f"Processing category '{category}' in '{set_type}' set...")

        for video_dir in tqdm(os.listdir(category_dir), desc=f"Loading {category} videos in {set_type}"):
            json_path = os.path.join(category_dir, video_dir, f"{video_dir}.json")
            if os.path.isfile(json_path):
                sequences = load_json_data(json_path, label)
                if sequences:
                    X.extend(sequences)
                    y.extend([label] * len(sequences))
                else:
                    logging.info(f"No data extracted from {json_path}")

# Ensure X and y have data
if not X or not y:
    logging.error("No data loaded. Please check dataset path and JSON structure.")
else:
    # Convert to numpy arrays and reshape for LSTM input
    X = np.array(X)
    y = np.array(y)

    # Log the shapes of X and y
    logging.info(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

    # Reshape X to fit LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], no_of_timesteps, len(keypoint_labels) * 2))

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log the shape of training and testing sets
    logging.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# %%
import os
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import datetime as dt

# Set up logging
logging.basicConfig(level=logging.INFO)

# %%
# Parameters
root_dir = os.path.join(os.path.dirname(os.getcwd()), "dataset_processing", "archive", "keypoints-rwf-2000")
no_of_timesteps = 20
keypoint_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
feature_dim = len(keypoint_labels) * 2  # x and y coordinates

# Initialize dataset lists
X = []
y = []

# %%
# Custom callback for live plotting
class LivePlotCallback(Callback):
    def __init__(self, save_path="models/training_progress.png"):
        super().__init__()
        self.save_path = save_path

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])
        self.accuracies.append(logs["accuracy"])
        self.val_accuracies.append(logs["val_accuracy"])

        self.ax[0].cla()
        self.ax[0].plot(self.losses, label="Training Loss", color="blue")
        self.ax[0].plot(self.val_losses, label="Validation Loss", color="orange")
        self.ax[0].set_title("Loss")
        self.ax[0].legend()

        self.ax[1].cla()
        self.ax[1].plot(self.accuracies, label="Training Accuracy", color="green")
        self.ax[1].plot(self.val_accuracies, label="Validation Accuracy", color="red")
        self.ax[1].set_title("Accuracy")
        self.ax[1].legend()

        plt.pause(0.01)
        plt.draw()
        self.fig.savefig(self.save_path)  # Save plot to file at the end of each epoch

    def on_train_end(self, logs=None):
        plt.ioff()
        # No need to call plt.show()

# %%
def load_json_data(json_path, label):
    """Load keypoints from a JSON file and process them into sequences."""
    try:
        with open(json_path) as file:
            data = json.load(file)
            if len(data) < no_of_timesteps:
                logging.warning(f"Skipping {json_path} as it has fewer than {no_of_timesteps} frames.")
                return None

            sequences = []
            for i in range(no_of_timesteps, len(data)):
                sequence = []
                frames = data[i - no_of_timesteps:i]

                for frame in frames:
                    if frame.get("detections"):
                        person = frame["detections"][0]
                        person_keypoints = []
                        keypoints_dict = {kp['label']: kp['coordinates'] for kp in person['keypoints']}
                        
                        for label in keypoint_labels:
                            if label in keypoints_dict:
                                coords = keypoints_dict[label]
                                person_keypoints.extend([coords['x'], coords['y']])
                            else:
                                person_keypoints.extend([0.0, 0.0])
                    else:
                        person_keypoints = [0.0, 0.0] * len(keypoint_labels)
                    
                    sequence.append(person_keypoints)
                sequences.append(sequence)

            return np.array(sequences)

    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        return None

# %%
def process_dataset(root_dir):
    """Load and process the dataset."""
    global X, y
    for category in ['train', 'val']:
        for label in ['Fight', 'NonFight']:
            category_dir = os.path.join(root_dir, category, label)
            logging.info(f"Processing category '{label}' in '{category}' set...")

            for video_folder in tqdm(os.listdir(category_dir)):
                video_folder_path = os.path.join(category_dir, video_folder)

                if os.path.isdir(video_folder_path):
                    json_path = os.path.join(video_folder_path, f"{video_folder}.json")

                    if os.path.isfile(json_path):
                        sequences = load_json_data(json_path, label)
                        if sequences is not None:
                            X.extend(sequences)
                            y.extend([1 if label == 'Fight' else 0] * len(sequences))

# %%
def build_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# %%
# Load dataset
logging.info("Loading dataset...")
process_dataset(root_dir)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Normalize data
mean = X.mean(axis=(0, 1))
std = X.std(axis=(0, 1))
X = (X - mean) / std

# Save normalization parameters
np.save("models/mean.npy", mean)
np.save("models/std.npy", std)

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# %%
# Train model
logging.info("Training model...")
input_shape = (no_of_timesteps, feature_dim)
model = build_model(input_shape)

callbacks = [
    LivePlotCallback(),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("models/violence_detector_best.keras", monitor="val_loss", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

# %%
# Evaluate model
logging.info("Evaluating model...")
loss, accuracy = model.evaluate(X_val, y_val)
logging.info(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")


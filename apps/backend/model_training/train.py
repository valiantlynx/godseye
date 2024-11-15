# %%
import os
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping
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

# Initialize dataset lists
X = []
y = []

# %%
# Custom callback for live plotting
class LivePlotCallback(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracies = []
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))
        self.ax[0].set_title("Loss")
        self.ax[1].set_title("Accuracy")

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        self.accuracies.append(logs["accuracy"])
        
        # Clear and update loss plot
        self.ax[0].cla()
        self.ax[0].plot(self.losses, label="Training Loss", color="blue")
        self.ax[0].set_title("Loss")
        self.ax[0].legend()

        # Clear and update accuracy plot
        self.ax[1].cla()
        self.ax[1].plot(self.accuracies, label="Training Accuracy", color="green")
        self.ax[1].set_title("Accuracy")
        self.ax[1].legend()
        
        plt.pause(0.01)  # Small pause to update the plot
        plt.draw()

    def on_train_end(self, logs=None):
        plt.ioff()
        plt.show()

# %%
def load_json_data(json_path, label):
    try:
        with open(json_path) as file:
            data = json.load(file)
            frames_data = []

            if len(data) < no_of_timesteps:
                logging.warning(f"Skipping {json_path} as it has fewer than {no_of_timesteps} frames.")
                return None

            for i in range(no_of_timesteps, len(data)):
                sequence = []
                frames = data[i - no_of_timesteps:i]

                for frame in frames:
                    if frame["detections"]:
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

                frames_data.append(np.array(sequence))

            return frames_data

    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        return None

# %%
def process_dataset(root_dir):
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

                        if sequences:
                            X.extend(sequences)
                            y.extend([1 if label == 'Fight' else 0] * len(sequences))


# Load and process dataset
process_dataset(root_dir)

# %%
# Convert to numpy arrays with correct shape
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)


print("Dataset shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

if len(X) == 0:
    raise ValueError("No data was loaded. Check the dataset directory and file paths.")

# Normalize the coordinates
mean = np.mean(X.reshape(-1, X.shape[-1]), axis=0)
std = np.std(X.reshape(-1, X.shape[-1]), axis=0)
std = np.where(std == 0, 1, std)
X = (X - mean) / std

# %%
# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# %%
# Model definition
model = Sequential([
    LSTM(64, input_shape=(no_of_timesteps, len(keypoint_labels) * 2), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

epochs=100
# Train the model with the LivePlotCallback
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks = [early_stopping_callback])


model_evaluation_history = model.evaluate(X_test, y_test)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

model_file_name = f'skeletonViolenceLSTM_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}__Epochs_{epochs}.h5'
model_path = os.path.join('models', model_file_name)

model.save(model_path)



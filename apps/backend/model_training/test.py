import os
import json
import numpy as np
from keras.models import load_model
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Parameters (same as training)
no_of_timesteps = 20
keypoint_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class ViolenceDetector:
    def __init__(self, model_path="models/all60/Keypoints_total.keras", mean_path="models/all60/Keypoints_total_mean.npy", std_path="models/all60/Keypoints_total_std.npy"):
        print(model_path)
        """Initialize the violence detector with a trained model and normalization parameters."""
        self.model = load_model(model_path)
        logging.info(f"Loaded model from {model_path}")
        
        # Load normalization parameters
        try:
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            logging.info(f"Loaded normalization parameters: mean from {mean_path}, std from {std_path}")
        except Exception as e:
            logging.warning(f"Could not load normalization parameters: {e}")
            self.mean = np.zeros(len(keypoint_labels) * 2)
            self.std = np.ones(len(keypoint_labels) * 2)

    def process_keypoints(self, json_data):
        """Process keypoints from JSON data into sliding window sequences."""
        try:
            if len(json_data) < no_of_timesteps:
                logging.warning(f"Not enough frames: Required {no_of_timesteps}, Got {len(json_data)}")
                return None

            sequences = []
            for i in range(no_of_timesteps, len(json_data) + 1):
                sequence = []
                frames = json_data[i - no_of_timesteps:i]
                
                for frame in frames:
                    person_keypoints = []
                    if frame.get("detections"):
                        # Get the first person's keypoints
                        person = frame["detections"][0]
                        keypoints_dict = {kp['label']: kp['coordinates'] for kp in person['keypoints']}
                        
                        # Extract coordinates in order
                        for label in keypoint_labels:
                            if label in keypoints_dict:
                                coords = keypoints_dict[label]
                                person_keypoints.extend([coords['x'], coords['y']])
                            else:
                                person_keypoints.extend([0.0, 0.0])
                    else:
                        # Fill with zeros if no detections
                        person_keypoints = [0.0, 0.0] * len(keypoint_labels)

                    sequence.append(person_keypoints)

                sequences.append(sequence)

            # Convert to numpy array and normalize
            sequences = np.array(sequences, dtype=np.float32)
            normalized_sequences = (sequences - self.mean) / self.std
            
            return normalized_sequences

        except Exception as e:
            logging.error(f"Error processing keypoints: {e}")
            return None

    def predict(self, json_data):
        """
        Predict violence probability from keypoints data.

        Args:
            json_data: List of frames with keypoint detections.

        Returns:
            dict: Prediction results including probability and classification.
        """
        # Process the keypoints into sequences
        sequences = self.process_keypoints(json_data)
        if sequences is None:
            return {
                "error": "Failed to process input data",
                "probability": None,
                "is_violent": None
            }

        # Make prediction
        try:
            pred_probs = self.model.predict(sequences, verbose=0)
            avg_prob = float(np.mean(pred_probs))  # Average probability across sequences
            
            return {
                "probability": avg_prob,
                "is_violent": avg_prob > 0.95,  # Adjust threshold if needed
                "confidence": max(avg_prob, 1 - avg_prob)
            }
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "probability": None,
                "is_violent": None
            }


def evaluate_validation_dataset(model_path, val_dir, mean_path="models/mean.npy", std_path="models/std.npy"):
    """Evaluate the model on a validation dataset."""
    detector = ViolenceDetector(model_path, mean_path, std_path)

    video_predictions = []
    video_labels = []
    processed_files = 0
    failed_files = 0
    
    # Process Fight and NonFight videos
    fight_dir = os.path.join(val_dir, "Fight")
    nonfight_dir = os.path.join(val_dir, "NonFight")
    
    logging.info("Processing validation dataset...")
    
    def process_directory(directory, label):
        print(directory)
        nonlocal processed_files, failed_files, video_predictions, video_labels
        
        for video_folder in tqdm(os.listdir(directory), desc=f"Processing {os.path.basename(directory)}"):
            video_folder_path = os.path.join(directory, video_folder)
            if os.path.isdir(video_folder_path):
                json_path = os.path.join(video_folder_path, f"{video_folder}.json")
                
                if os.path.isfile(json_path):
                    try:
                        with open(json_path) as f:
                            json_data = json.load(f)
                        
                        result = detector.predict(json_data)
                        if result["error"] is None:
                            video_predictions.append(result["is_violent"])
                            video_labels.append(label)
                            processed_files += 1
                        else:
                            failed_files += 1
                    except Exception as e:
                        logging.error(f"Failed to process {json_path}: {e}")
                        failed_files += 1

    # Process both directories
    process_directory(fight_dir, 1)
    process_directory(nonfight_dir, 0)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    if video_predictions and video_labels:
        acc = accuracy_score(video_labels, video_predictions)
        prec = precision_score(video_labels, video_predictions)
        rec = recall_score(video_labels, video_predictions)
        f1 = f1_score(video_labels, video_predictions)
        cm = confusion_matrix(video_labels, video_predictions)
        
        logging.info(f"Processed {processed_files} files with {failed_files} failures")
        logging.info(f"Accuracy: {acc:.2%}, Precision: {prec:.2%}, Recall: {rec:.2%}, F1 Score: {f1:.2%}")
        logging.info(f"Confusion Matrix:\n{cm}")
    else:
        logging.warning("No valid predictions to calculate metrics.")


if __name__ == "__main__":
    model_path = "models/violence_detector_best.keras"
    val_dir = "C:/Users/gorme/projects/godseye/apps/backend/dataset_processing/archive/keypoints-rwf-2000/val"
    mean_path = "models/mean.npy"
    std_path = "models/std.npy"

    evaluate_validation_dataset(model_path, val_dir, mean_path, std_path)

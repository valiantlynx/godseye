import os
import json
import numpy as np
from keras.models import load_model
import logging

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
    def __init__(self, model_path="violence_detector_best.keras", mean_path="mean.npy", std_path="std.npy"):
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
                "is_violent": avg_prob > 0.5,  # Adjust threshold if needed
                "confidence": max(avg_prob, 1 - avg_prob)
            }
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "probability": None,
                "is_violent": None
            }

# Example usage
def main():
    # Load the detector
    detector = ViolenceDetector()

    # Example of how to use it with a JSON file
    def process_video_json(json_path):
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            result = detector.predict(json_data)
            logging.info(f"Prediction result: {result}")
        except Exception as e:
            logging.error(f"Error processing video JSON: {e}")

    # Example JSON path
    json_path = "path_to_video_json/video.json"
    process_video_json(json_path)

if __name__ == "__main__":
    main()

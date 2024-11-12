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
    def __init__(self, model_path="models/acc_96__loss_0.1__Epochs_30.h5"):
        """Initialize the violence detector with a trained model."""
        self.model = load_model(model_path)
        
        # Load normalization parameters (you'll need to save these during training)
        # For now, we'll use placeholder values - you should save and load actual values
        self.mean = np.zeros(34)  # Replace with actual mean values from training
        self.std = np.ones(34)    # Replace with actual std values from training
        
    def process_keypoints(self, json_data):
        """Process keypoints from JSON data into model input format."""
        try:
            frames_data = []
            
            # Ensure we have enough frames
            if len(json_data) < no_of_timesteps:
                logging.warning(f"Not enough frames. Required: {no_of_timesteps}, Got: {len(json_data)}")
                return None
            
            # Process each frame in the sequence
            sequence = []
            for frame in json_data[-no_of_timesteps:]:  # Take last no_of_timesteps frames
                person_keypoints = []
                
                if frame["detections"]:
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
                    # If no detections, fill with zeros
                    person_keypoints = [0.0, 0.0] * len(keypoint_labels)
                
                sequence.append(person_keypoints)
            
            # Convert to numpy array and normalize
            sequence = np.array(sequence)
            sequence = (sequence - self.mean) / self.std
            
            return np.expand_dims(sequence, axis=0)  # Add batch dimension
            
        except Exception as e:
            logging.error(f"Error processing keypoints: {e}")
            return None
    
    def predict(self, json_data):
        """
        Predict violence probability from keypoints data.
        
        Args:
            json_data: List of frames with keypoint detections
            
        Returns:
            dict: Prediction results including probability and classification
        """
        # Process the keypoints
        model_input = self.process_keypoints(json_data)
        
        if model_input is None:
            return {
                "error": "Failed to process input data",
                "probability": None,
                "is_violent": None
            }
        
        # Make prediction
        try:
            probability = float(self.model.predict(model_input)[0][0])
            
            return {
                "probability": probability,
                "is_violent": probability > 0.5,
                "confidence": max(probability, 1 - probability)
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
            
            # Get prediction
            result = detector.predict(json_data)
            
            # Print results
            if "error" not in result:
                print(f"/nResults for {json_path}:")
                print(f"Violence Probability: {result['probability']:.2%}")
                print(f"Classification: {'Violent' if result['is_violent'] else 'Non-violent'}")
                print(f"Confidence: {result['confidence']:.2%}")
            else:
                print(f"Error processing {json_path}: {result['error']}")
                
        except Exception as e:
            print(f"Error processing file {json_path}: {e}")

    # Example usage with a test file
    test_json_path = "C:/Users/gorme/projects/godseye/apps/backend/dataset_processing/archive/keypoints-rwf-2000/val/Fight/YDOJvzChqSg_2/YDOJvzChqSg_2.json"  # Replace with actual test file path
    if os.path.exists(test_json_path):
        process_video_json(test_json_path)
    else:
        print(f"Test file not found: {test_json_path}")

if __name__ == "__main__":
    main()
# godseye

This project implements a real-time violence detection system using **Pose Estimation** and **LSTM**. We use two benchmark datasets, **RWF-2000** and **Hockey Fight**, to train and evaluate the model. The project includes backend processing for pose extraction and model inference, as well as a frontend for live video monitoring.

 **Model Weights**: https://huggingface.co/valiantlynxz/godseye-violence-detection-model/tree/main
 
 **Dataset**: https://huggingface.co/datasets/valiantlynxz/godseye-violence-detection-dataset
 
## Table of Contents
- [godseye](#godseye)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Project Structure](#project-structure)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running Backend](#running-backend)
    - [Running Frontend](#running-frontend)
  - [Model Training](#model-training)
  - [Inference](#inference)
  - [Open Source](#open-source)
  - [Contributing](#contributing)
  - [License](#license)

## Overview
Our system identifies violent actions in videos by using pose keypoints extracted with **YOLO**. These keypoints are then passed to an **LSTM** model that predicts whether the actions in the video are violent or non-violent. 

### Project Structure
```

/godseye
│
├── /apps
│   ├── /backend
│   │   ├── /dataset_processing       # Scripts to preprocess datasets and extract keypoints
│   │   ├── /model_training           # Training scripts using TensorFlow
│   │   ├── /models                   # Trained models and checkpoints
|   |   |__ live_inference.py         # Inference scripts for real-time video feed processing
│   │   ├── requirements.txt          # Backend dependencies
│   │
│   ├── /frontend
│       ├── /static                   # Static files for UI
│       ├── /templates                # HTML/CSS for UI/UX
│       ├── /live_feed                # Frontend logic for live streaming and monitoring
│       ├── package.json              # Frontend dependencies (if using Node.js, React, etc.)
│
├── /docs                             # Documentation
│   └── README.md
│
├── README.md                         # Project overview and instructions
├── LICENSE                           # MIT License
└── .gitignore
```

## Dataset
We use two datasets for this project:
- **RWF-2000**: A surveillance footage dataset with 2000 videos.
- **Hockey Fight**: A sports video dataset containing 1000 clips of fighting and non-fighting.

We also preprocess these videos to create a new dataset with pose keypoints.

## Installation
To install dependencies for both backend and frontend, follow these steps:

```bash
# Clone the repository
git clone https://github.com/valiantlynx/godseye.git
cd godseye

# Backend dependencies
cd apps/backend

## start environment use your favorite ide, conda, etc. otherwise use python venv
python -m venv .venv
source .venv/bin/activate # if windows .\.venv\Scripts\acticate

pip install -r requirements.txt

# Frontend dependencies (if using a framework like React)
cd apps/frontend
npm install
```

## Usage

### Running Backend
1. **Dataset Preparation**: Extract keypoints from the dataset using:
    ```bash
    python backend/dataset_processing/extract_keypoints.py
    ```
2. **Model Training**: Train the LSTM model using:
    ```bash
    python backend/model_training/train.py
    ```

3. **Inference**: To run inference on a live video stream:
    ```bash
    python backend/inference/live_inference.py
    ```

### Running Frontend
To start the frontend for live feed monitoring:
```bash
cd apps/frontend
npm start
```

## Model Training
The LSTM model is trained on pose keypoints extracted from the videos. You can modify the training script in the `/apps/backend/model_training` directory to tune hyperparameters and model architecture.

## Inference
Inference processes pose keypoints in real-time, feeding them to the trained LSTM model to predict violent or non-violent behavior.

## Open Source
- The trained model will be uploaded to **Hugging Face**.
- The processed dataset will be shared on **Kaggle**.

## Contributing
Feel free to fork this repository, submit issues, or send pull requests for improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.


git clone ssh://git@tools.uia.no:7999/ikt213g24h/godseye.git

model and dataset: https://huggingface.co/valiantlynxz/convlstm_model
clone it to the root of this project


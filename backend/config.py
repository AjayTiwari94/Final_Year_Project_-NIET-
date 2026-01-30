import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Datasets"
MODEL_DIR = BASE_DIR / "backend" / "saved_models"
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"

# Dataset paths
APTOS_DIR = DATASET_DIR / "APTOS_dataset"
HAM10000_DIR = DATASET_DIR / "HAM10000_dataset"
MURA_DIR = DATASET_DIR / "MURA_dataset" / "MURA-v1.1"

# Model parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

# Class mappings
APTOS_CLASSES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

HAM10000_CLASSES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

MURA_CLASSES = {
    0: "Normal",
    1: "Abnormal/Fracture"
}

MURA_BODY_PARTS = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

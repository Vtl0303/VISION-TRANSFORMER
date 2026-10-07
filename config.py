"""
Configuration file for Deepfake vs Real Faces Detection using ViT
"""

import os
from pathlib import Path

# Dataset paths
DATASET_PATH = Path('Dataset/')
TRAIN_PATH = DATASET_PATH / 'Train'
TEST_PATH = DATASET_PATH / 'Test'
VALIDATION_PATH = DATASET_PATH / 'Validation'

# Model configuration
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
OUTPUT_DIR = "deepfake_vs_real_image_detection"
LOGS_DIR = "./logs"

# Training parameters
NUM_EPOCHS = 2
LEARNING_RATE = 1e-6
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.02
WARMUP_STEPS = 50
TEST_SIZE = 0.4
RANDOM_STATE = 83

# Image processing
IMAGE_SIZE = 224
RANDOM_ROTATION = 90
SHARPNESS_FACTOR = 2

# Labels
LABELS_LIST = ['Real', 'Fake']

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"

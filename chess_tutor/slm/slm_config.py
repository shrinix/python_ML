MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "./chess_tutor_slm_model"

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 200

TEMPERATURE_PRIMARY = 0.4
TEMPERATURE_RETRY = 0.2

MAX_RETRIES = 2

# Path to training data folder (can be overridden)
TRAIN_DATA_FOLDER = "./chess_tutor/slm/data"


# Training hyperparameters
TRAIN_NUM_EPOCHS = 1  # For quick e2e test
TRAIN_BATCH_SIZE = 1
TRAIN_GRAD_ACCUM = 1
TRAIN_LEARNING_RATE = 2e-4

# Limit number of training samples (None for all, or set to an int for fast test)
TRAIN_SAMPLE_LIMIT = 10

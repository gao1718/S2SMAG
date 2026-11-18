"""Network and optimizer settings for the C1/C9 reproduction."""

from pathlib import Path

from model.config_base import BaseConfig


class Config(BaseConfig):
    # Run settings
    SCRIPT_DIR = Path(__file__).resolve().parent
    MODEL_NAME = "S2SMAG"
    VARIANT = "s2smag"
    FOLD = 0
    EXPERIMENT_NAME = "c1_test_c9_validation"
    SMOKE = False

    # Settlement, settlement rate, and vacuum pressure. Only vacuum is given in the forecast period.
    INPUT_SIZE = 3
    FUTURE_INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    ENCODER_LENGTH = 10
    HORIZON = 10

    # BiGRU encoder, multi-head cross-attention, and GRU decoder
    HIDDEN_DIM = 432
    DECODER_HIDDEN_DIM = 352
    NUM_LAYERS = 2
    DECODER_NUM_LAYERS = 3
    ENCODER_DROPOUT = 0.2
    DECODER_DROPOUT_ENABLED = True
    TEACHER_FORCING_RNG = "python"
    DROPOUT = 0.1
    ATTENTION_HEADS = 2
    TEACHER_FORCING_RATIO = 0.8  # Training only

    # Optimization
    EPOCHS = 1000
    BATCH_SIZE = 16
    LEARNING_RATE = 9.628021160532742e-05
    BETAS = (0.9, 0.99)
    EPS = 1e-8
    WEIGHT_DECAY = 0.001
    LOOKAHEAD_K = 2
    LOOKAHEAD_ALPHA = 0.7
    SCHEDULER_ENABLED = False
    SCHEDULER_T_0 = 10000
    SCHEDULER_T_MULT = 1
    MIN_LEARNING_RATE = 1e-5
    GRADIENT_CLIP_NORM = None

    # Reproducibility and logging
    RANDOM_SEED = 42
    SHUFFLE_TRAINING = True
    DEVICE = "auto"
    CPU_THREADS = 2
    LOG_INTERVAL = 10
    CHECKPOINT_INTERVAL = 100
    BEST_MODEL_FILENAME = "S2SMAG_best.pth"
    FINAL_MODEL_FILENAME = "S2SMAG_final.pth"


cfg = Config()

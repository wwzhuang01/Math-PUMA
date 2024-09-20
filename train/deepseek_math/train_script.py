import os
import subprocess
from pathlib import Path
from datetime import datetime

SEED = int(os.environ.get("SEED", 3407))

time_str = datetime.now().strftime("%m%d_%H")
run_name = os.environ.get("RUN_NAME", "mathpuma")
RUN_NAME = f"R_{run_name}_D_{time_str}"


######################################### PATH and DIRECTORIES #########################################

MODEL_PATH = Path(os.environ.get("ORIGIN_MODEL_PATH", "/mnt/bn/gmma/models/deepseek-math-vl-aligned"))
MODEL_NAME = MODEL_PATH.name
MODEL_SAVE_DIR = Path(os.environ.get("MODEL_SAVE_DIR", "./temp"))
MODEL_SAVE_PATH = MODEL_SAVE_DIR / RUN_NAME

# Support for loading multiple datasets of the same type, separated by commas
DATA_PATH = os.environ.get("DATA_PATH",
    "/mnt/bn/gmma/datasets/VarsityTutors/caption/with_image_cot_kd.jsonl"
)

######################################### TRAINING ARGUMENTS #########################################
NUM_EPOCHS = os.environ.get("NUM_EPOCHS", "1")
PER_DEVICE_BATCH_SIZE = int(os.environ.get("PER_DEVICE_BATCH_SIZE", 1))
GRAD_ACC_STEPS = int(os.environ.get("GRAD_ACC_STEPS", 16))

OPTIM = str(os.environ.get("OPTIM", "adamw_torch"))
BF16 = str(os.environ.get("BF16", "true"))
FP16 = str(os.environ.get("FP16", "false"))
TF32 = str(os.environ.get("TF32", "true"))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-5))
LR_SCHEDULER_TYPE = str(os.environ.get("LR_SCHEDULER_TYPE", "cosine"))

WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.01))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.1))
MODEL_MAX_LENGTH = int(os.environ.get("MODEL_MAX_LENGTH", 2048))

TRAINABLE_PARTS = str(os.environ.get(   # "aligner, language_model, vision_tower_low, vision_tower_high"
    "TRAINABLE_PARTS", "aligner"
))

# KL
USE_KL = str(os.environ.get("USE_KL", "true")) 
ALPHA_KL = float(os.environ.get("ALPHA_KL", 0.5))
LAMBDA_KL = float(os.environ.get("LAMBDA_KL", 0.1))
TEMP_KL = float(os.environ.get("TEMP_KL", 1.0))

LAZY_LOAD = str(os.environ.get("LAZY_LOAD", "true"))
SHUFFLE_DATA = str(os.environ.get("SHUFFLE_DATA", "true"))

SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 1000))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", 2))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", 1))


######################################### ENVIRONMENT VARIABLES #########################################
os.environ["WANDB_NAME"] = RUN_NAME
os.environ["WANDB_PROJECT"] = "mathpuma"

launcher_command = (
    f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" TOKENIZERS_PARALLELISM=false"
    f" torchrun"
    # f" --master_addr <master_addr>"
    # f" --master_port <master_port>"
    # f" --nnodes <nnodes>"
    # f" --node_rank <node_rank>"
    # f" --nproc_per_node <nproc_per_node>"
    f" ./train/deepseek_math/train_main.py "
)

train_args = (
    f" --model_name_or_path {MODEL_PATH}"
    f" --model_max_length {MODEL_MAX_LENGTH}"
    f" --data_path {DATA_PATH}"
    f" --output_dir {MODEL_SAVE_PATH}"
    f" --num_train_epochs {NUM_EPOCHS}"
    f" --per_device_train_batch_size {PER_DEVICE_BATCH_SIZE}"
    f" --gradient_accumulation_steps {GRAD_ACC_STEPS} "
    f" --learning_rate {LEARNING_RATE}"
    f" --lr_scheduler_type '{LR_SCHEDULER_TYPE}'"
    f" --optim {OPTIM}"
    f" --bf16 {BF16}"
    f" --fp16 {FP16}"
    f" --tf32 {TF32}"
    f" --weight_decay {WEIGHT_DECAY}"
    f" --warmup_ratio {WARMUP_RATIO}"
    f" --trainable_parts {TRAINABLE_PARTS}"
    f" --use_kl {USE_KL}"
    f" --alpha_kl {ALPHA_KL}"
    f" --lambda_kl {LAMBDA_KL}"
    f" --temperature_kl {TEMP_KL}"
    f" --lazy_load {LAZY_LOAD}"
    f" --shuffle_data {SHUFFLE_DATA}"
    f" --save_strategy 'steps'"
    f" --save_steps {SAVE_STEPS}"
    f" --save_total_limit {SAVE_TOTAL_LIMIT}"
    f" --logging_steps {LOGGING_STEPS}"
    f" --report_to 'wandb'"
    f" --fsdp 'full_shard auto_wrap'"
    f" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'"
    f" --seed {SEED}"
)


train_command = f"{launcher_command} {train_args}"

subprocess.run(train_command, shell=True)

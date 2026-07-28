#!/bin/bash

# Find the path where THIS script is
SCRIPT_DIR=$(dirname "$0")  # NO SPACE AROUND '=' (bash syntax)
                            # $0 is a special command containing always the exact path to use to launch the script
                            # e.g. if you need to do 'bash lora/script.sh' $0 contains 'lora/script.sh'
                            # 'dirname' extracts the name of the directory (e.g. lora/script.sh --> lora)
                            # $() is *command substitution*: the program is paused, the command inside () is executed and the resulting text is assigned via '='

# Define paths based on the current location 
export MODEL_NAME="runwayml/stable-diffusion-v1-5" # 'export' creates envioronmental variables, accessible by any branch of the child process
export DATASET_NAME="$SCRIPT_DIR/gibli_lora_dataset" 
export OUTPUT_DIR="$SCRIPT_DIR/lora_ghibli_model" # where the result will be 

accelerate launch train_text_to_image_lora.py \ # here no extra info because the ft script is at the same level of THIS
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --learning_rate=1e-04 \
  --output_dir=$OUTPUT_DIR
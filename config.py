from pathlib import Path

# ROOT = Path.cwd() --- works only if THIS script is run from THIS folder
ROOT = Path(__file__).resolve().parent # absolute path pointing always to the position of THIS file wherever the script is run. All other paths are correct once this is correct.
DEST_PATH = ROOT / "generated_images" 
LORA_PATH = ROOT / "models" / "pytorch_lora_weights.safetensors"
POSITIVE_PROMPT = "studio ghibli style, masterpiece, best quality, ultra-detailed"
NEGATIVE_PROMPT = "ugly, gross, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, worst quality, low quality, deformed face, mutated, poorly drawn, deformed"
MAX_ITER = 3
DIFF_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# LM_NAME = "Qwen/Qwen2.5-7B-Instruct"
LM_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # this is for test 
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"



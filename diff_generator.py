from PIL import Image
from config import NEGATIVE_PROMPT, LORA_PATH, REPO_ID, SAFETENSORS_FILENAME
from diffusers import StableDiffusionPipeline 
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_safetensors_from_hf(repo_id: str, filename: str, local_dir: Path) -> Path:
  """
  Downloads a .safetensors file from a Hugging Face model repository.
  """
  local_dir.mkdir(parents=True, exist_ok=True)
  downloaded_path = hf_hub_download(
      repo_id=repo_id,
      filename=filename,
      local_dir=local_dir
  )
  print(f"Downloaded '{filename}' to {downloaded_path}")
  return Path(downloaded_path)



class Img_Generator(): 
    def __init__(self, base_model:str):
        """Initialize the model and load lora weights."""

        download_safetensors_from_hf(
            repo_id=REPO_ID,
            filename=SAFETENSORS_FILENAME,
            local_dir=LORA_PATH.parent
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
              base_model,
              torch_dtype=torch.float16
        ) # .to("cuda")
        self.pipe.load_lora_weights(LORA_PATH)
        self.pipe.enable_model_cpu_offload()    # STABLE DIFFUSION = CLIP(text encoder) --> UNET(encoder) --> DIFFUSION PROCESS IN THE BOTTLENECK --> VAE(decoder)  
                                                # ---> move on the GPU one per time 

    @torch.no_grad()    # inference mode <=> no need to store gradientes <=> spare VRAM
    def generate(self, prompt:str) -> Image: 
        """Generate an image."""
        img: Image = self.pipe(prompt=prompt, negative_prompt=NEGATIVE_PROMPT).images[0]
        return img        
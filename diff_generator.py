from PIL import Image
from config import NEGATIVE_PROMPT, LORA_PATH
from diffusers import StableDiffusionPipeline 
import torch

class Img_Generator(): 
    def __init__(self, base_model:str):
        """Initialize the model and load lora weights."""
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
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from typing import Literal

# ATTEMPT TO HAVE FULL CPU MODEL DUE TO HW LIMITATION OF MY MACHINE

class CLIP_Evaluator:
    def __init__(self, clip_model:str):
        self.model = CLIPModel.from_pretrained(
            clip_model, 
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        #.to('cpu') # Why to use safetensors? ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
        self.processor = CLIPProcessor.from_pretrained(clip_model)

    @torch.no_grad()
    def get_score(self, img:Image, text:str) -> float: 
        """Return cosine similarity between img and text embeddings."""

        # self.model.to('cuda')   # Here it's the model to be moved to the GPU

        inputs = self.processor(
            text=text,
            images=img,
            return_tensors="pt",
            padding=True
        ) # Here it's the inputs (numerical values) to be moved to the GPU

        outputs = self.model(**inputs)
        score = torch.cosine_similarity(outputs.text_embeds, outputs.image_embeds)

        # move back
        # self.model.to('cpu')

        return score.item() 

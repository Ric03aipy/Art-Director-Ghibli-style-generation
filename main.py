from diff_generator import Img_Generator
from prompt_optimizer import Prompt_Optimizer
from evaluator import CLIP_Evaluator
from config import *

from PIL import Image
from pathlib import Path
import torch

import gc # forcing to save memory

import gradio as gr

# GLOBALS: model initialization - to do once
img_gen = Img_Generator(DIFF_MODEL_NAME)    # this is moved automatically on the CPU and GPU by enabling the offload
prompt_opt = Prompt_Optimizer(LM_NAME)      # this must be on the GPU due to quantization, but memory consumption is drastically reduced by quantization
evaluator = CLIP_Evaluator(CLIP_MODEL_NAME)



def generate_art(user_prompt:str) -> tuple[Image, str, float]: 
    """Encapsulate the whole logics. From a prompt to all generative products."""
    best_score = -1 # minimum cosine similarity

    best_prompt = None
    best_img = None

    # run MAX_ITER times the core logic
    for i in range(MAX_ITER): 
        refined_prompt = prompt_opt.enhance(user_prompt)
        img: Image = img_gen.generate(refined_prompt)
        score = evaluator.get_score(img, refined_prompt)

        # update 
        if score > best_score: 
            best_score = score
            best_img = img
            best_prompt = refined_prompt

        # recollecting some memory
        if img != best_img: 
            del img
        gc.collect()
        torch.cuda.empty_cache()
    
    # import numpy as np
    # fake_image = np.zeros((224, 224, 3), dtype=np.uint8)
    # fake_image[100:200, 100:200, :] = 255
    # img = Image.fromarray(fake_image)
    # best_img = img
    # best_prompt = "something good"
    # best_score = 0.3
    return (best_img, best_prompt, best_score)

def save_img(img:Image, name:str="result") -> None:
    img.save(DEST_PATH / f"{name}.png")



def main():
    # setup 
    Path.mkdir(DEST_PATH, parents=True, exist_ok=True)

    # gradio interface
    with gr.Blocks(title="Multimodal art director") as demo: 
        with gr.Row(): # row with the text to insert
            inputs = gr.TextArea(label="User idea")
            gen_btn = gr.Button(value="Generate")
        with gr.Row(): # img
            img = gr.Image(label="Generated image", type="pil", interactive=False)
        with gr.Row(): # additional outputs
            refined_prompt = gr.TextArea(label="Enchanced prompt")
            score = gr.Textbox(label="CLIP score")
        with gr.Row(): 
            name = gr.Textbox(value="result", label="Name the file to save as...")
            save_btn = gr.Button(value="Save")

        gen_btn.click(
            generate_art,
            inputs=inputs,
            outputs=[img, refined_prompt, score]
        )
        save_btn.click(
            save_img, 
            inputs=[img, name],
            outputs=[]
        )

    demo.launch()



if __name__ == "__main__":
    main()

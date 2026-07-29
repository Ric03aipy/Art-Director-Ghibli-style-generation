# 🎨 Multimodal Art Director: End-to-End GenAI Pipeline

## 📖 Overview
A lightweight, fully containerized AI pipeline that acts as a virtual Art Director. It takes a simple user idea - in the form of a prompt -, enhances it using a quantized Large Language Model, generates the artwork using Stable Diffusion with a custom LoRA - to transfer Ghibli Studio style -, and evaluates the output quality using CLIP in a "Best-of-N" loop.

## 🏗️ System Architecture
This project orchestrates multiple AI models to work seamlessly together:
1. **Semantic Brain (LLM):** Qwen (4-bit NF4 quantized) expands a basic user prompt into a highly detailed, professional diffusion prompt.
2. **Visual Engine (Diffusion):** Stable Diffusion v1.5 injected with a custom Studio Ghibli LoRA generates the images.
3. **Quality Evaluator (VLM):** CLIP calculates the cosine similarity between the generated images and the prompt, selecting the best one out of N iterations.
4. **Interface:** A simple Web UI built with Gradio.

## ⚙️ Hardware Optimization (Built for 4GB VRAM)
This pipeline is heavily engineered to run on highly constrained hardware (like a 4GB GPU on WSL) without triggering CUDA Out-Of-Memory errors:
- **LLM Quantization:** The LLM is compressed using `bitsandbytes` (NF4, double quantization) to drastically reduce VRAM footprint.
- **CPU Offloading:** The Stable Diffusion pipeline leverages Hugging Face's `enable_model_cpu_offload()`, moving sub-components (Text Encoder, U-Net, VAE) to RAM when not actively computing.
- **VLM Isolation:** The CLIP evaluator is deliberately confined to the CPU to prevent VRAM fragmentation during the Best-of-N loop.
- **Manual Garbage Collection:** Explicit `gc.collect()` and `torch.cuda.empty_cache()` are triggered after every iteration.

If more resources are available you can simply tweak global variables in the file `config.py`.

If the user wants to upload his `.safetensors` file he can load it on Hugging Face models and just change global variables in the file `config.py`. 

## 🚀 How to Run (Production via Docker)

**1. Clone the repository**
```bash
git clone https://github.com/Ric03aipy/Art-Director-Ghibli-style-generation.git
```
**2. Change to main directory**
```
cd Art-Director-Ghibli-style-generation
```
**3. Build the image**
```
docker build -t art_director .
```
**4. Run the container**
```
docker run -p 7860:7860 --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v $(pwd)/models:/app/models -v $(pwd)/output:/app/generated_images art_director
```

Note that first run can take a while to download and install all the dependencies. 

**5. Open the web UI**

Connect to `http://127.0.0.1:7860/`. 

You will see a simple web UI made with Gradio. 

Write your idea and click 'generate' to start the flow. 

After generation, you can name the file and save it. It will be available in the folder `output/`
# This doesn't support cuda/torch natively
# FROM python:3.12.13-bookworm

# This image is provided directly with the torch+cuda version used in the project
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/

# --no-cache-dir reduces the size of the image
RUN pip install --no-cache-dir -r requirements.txt

COPY    config.py \ 
        diff_generator.py \
        evaluator.py \
        main.py \
        prompt_optimizer.py \
        /app/

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "main.py"]



# in the folder with this file, build with:
# docker build -t art_director:2.0 .
# from the terminal launch: 
# docker run -p 7860:7860 --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v $(pwd)/models:/app/models -v $(pwd)/output:/app/generated_images art_director:2.0

# This way the user can substitute in his folder the .safetensors file and adapt to another style. 
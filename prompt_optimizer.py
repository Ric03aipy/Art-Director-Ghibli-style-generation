from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from config import POSITIVE_PROMPT

class Prompt_Optimizer: 
    def __init__(self, model_name:str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype="auto",
            device_map="auto",
            quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def enhance(self, prompt:str) -> str:
        messages = [
            {
                "role": "system", 
                "content": "Act as a professional prompt engineer for high quality image generation. You will receive simple ideas like 'A cat drinking a coffee' and you will have to transform into something like 'A majestic cat drinking a steaming cup of coffee in a lush green fantasy landscape, floating islands, studio ghibli style, intricate details, masterpiece, vibrant colors, clear sky'. **OUTPUT ONLY THE PROMPT**, No chatting attitude, no connectors, no introduction to the anwer, no final summary.",
            },
            {
                "role": "user", 
                "content": f"{prompt}"
            },
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True # since it's not clear when it needs and when it doesn't need to extract "input_ids" with this we force it to be a dict with "input_ids" as a key
        ).to(self.model.device) # pass to GPU
        input_ids = tokenized_chat["input_ids"]
        outputs = self.model.generate(
            input_ids, 
            max_new_tokens=150,
            temperature=0.5 # a bit of creativity
        )
        # print(outputs) --- all outputs (ids) 
        # print(outputs[0][input_ids.shape[-1]:]) --- only answer outputs (ids)
        # print(self.tokenizer.decode(outputs[0])) --- all chat + answer + special token 
        # print(self.tokenizer.decode(outputs[0][input_ids.shape[-1]:])) --- only answer + special token 

        refined_prompt = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        tuned_prompt = f"{refined_prompt}, {POSITIVE_PROMPT}"

        return tuned_prompt
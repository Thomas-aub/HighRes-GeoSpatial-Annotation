import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# Load the model and tokenizer
path = "ByteDance/Sa2VA-4B"
print("Loading model...")

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_flash_attn=True,
    low_cpu_mem_usage=True,
).eval().cuda()

print("Model loaded successfully!")

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Setup your image path
image_path = "data/ScreenshotBoat.png" 

# Prepare the input
text_prompts = "<image>Please describe how you find the boat in the image."

try:
    image = Image.open(image_path).convert('RGB')
except FileNotFoundError:
    print(f"Error: Could not find image at {image_path}")
    exit()

input_dict = {
    'image': image,
    'text': text_prompts,
    'past_text': '',
    'mask_prompts': None,
    'tokenizer': tokenizer,
}

# Run inference
print("Generating description...")
with torch.no_grad():
    return_dict = model.predict_forward(**input_dict)

answer = return_dict["prediction"] 

# Print the result
print("-" * 30)
print("Model Description:")
print(answer)
print("-" * 30)

import os
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline

# configuration parameters
DATASET_PATH = "dataset"
NUM_PAIRS = 100
NUM_INFERENCE_STEPS = 40
DEVICE = "cpu"  

PROMPT_CLEAN = (
    "Close-up portrait of a person with a symmetric clean-shaven face, "
    "highly detailed, photorealistic, focus on facial features, "
    "ultra HD, studio lighting, neutral background"
)
INSTRUCTION_BEARD = (
    "Add a realistic, well-groomed beard to this face while keeping "
    "all other facial features identical. Focus on the details of the face."
)
NEGATIVE_PROMPT = (
    "ugly, out of focus, blurry, deformed, surreal, cartoon, painting, text, watermark"
)

def create_dataset_folder(dataset_path: str) -> None:
    os.makedirs(dataset_path, exist_ok=True)

def initialize_pipelines(device: str = "cpu"):
    base_model_id = "runwayml/stable-diffusion-v1-5"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float32
    ).to(device)
    
    instruct_model_id = "timbrooks/instruct-pix2pix"
    ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        instruct_model_id, torch_dtype=torch.float32
    ).to(device)
    
    return sd_pipe, ip2p_pipe

def generate_image_pairs(sd_pipe, ip2p_pipe):
    create_dataset_folder(DATASET_PATH)
    for i in range(NUM_PAIRS):
        seed = np.random.randint(0, 1_000_000)
        generator_clean = torch.Generator(device="cpu").manual_seed(seed)
        clean_image = sd_pipe(
            prompt=PROMPT_CLEAN,
            negative_prompt=NEGATIVE_PROMPT,
            generator=generator_clean,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=512,
            width=512
        ).images[0]
        clean_image_path = os.path.join(DATASET_PATH, f"pair_{i:02d}_clean.png")
        clean_image.save(clean_image_path)
        
        generator_beard = torch.Generator(device="cpu").manual_seed(seed)
        beard_image = ip2p_pipe(
            prompt=INSTRUCTION_BEARD,
            image=clean_image,
            negative_prompt=NEGATIVE_PROMPT,
            generator=generator_beard,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=512,
            width=512
        ).images[0]
        beard_image_path = os.path.join(DATASET_PATH, f"pair_{i:02d}_beard.png")
        beard_image.save(beard_image_path)
        
        print(f"Generated pair: {clean_image_path} and {beard_image_path}")
        
    print(f"{NUM_PAIRS * 2} images (i.e., {NUM_PAIRS} pairs) have been created.")

if __name__ == "__main__":
    sd_pipe, ip2p_pipe = initialize_pipelines(device=DEVICE)
    generate_image_pairs(sd_pipe, ip2p_pipe)

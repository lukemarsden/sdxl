from diffusers import DiffusionPipeline
import torch
import os
import numpy as np
import random

# pip install --upgrade huggingface_hub
# huggingface-cli login --token $HUGGINGFACE_TOKEN

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(int(os.getenv("RANDOM_SEED", "42")))

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
#pipe.to("cuda")
pipe.enable_model_cpu_offload()

# Not on my puny 1080
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = os.getenv("PROMPT", "An astronaut riding a green horse")

images = pipe(prompt=prompt).images[0]

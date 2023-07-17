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

seed = int(os.getenv("RANDOM_SEED", "42"))
set_seed(seed)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

# For low GPU memory:
#pipe.enable_model_cpu_offload()

# To compile graph, but seems to be slower. Maybe the compiled graph isn't getting cached?
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = os.getenv("PROMPT", "An astronaut riding a green horse")

images = pipe(prompt=prompt).images
print(f"Got {len(images)} images")

image = images[0]

# OUTPUT_DIR must have a trailing slash
image.save(os.getenv("OUTPUT_DIR", "") + f"image-{seed}.png")

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from diffusers import AutoPipelineForImage2Image
import torch
from diffusers.utils import load_image, make_image_grid

from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipe = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
# init_image = load_image(url)
# prompt = "a dog catching a frisbee in the jungle"
# image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5).images[0]


lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
image = load_image("/home/ezalos/42/art/comfyui/swag_mirror/2023-12-31 22:33:11_in.png")

# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
prompt = "Mesmerizing oil painting of Santa Claus, by ((Henri de Toulouse-Lautrec)), by (Amedeo Modigliani), by (((Tom Thomson)))"
negative_prompt = "text, watermark, frames, border, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
image = load_image("/home/ezalos/42/art/comfyui/swag_mirror/2023-12-31 22:33:11_in.png")

import random
import PIL


def add_noise(x):
    x.save("test_in.png")
    noise = PIL.Image.effect_noise(x.size, sigma=50).convert("RGB")
    noise.save("test_rnd.png")
    print(f"{x.size = } {x.mode = }")
    print(f"{noise.size = } {noise.mode = }")
    img = PIL.Image.blend(x, noise, 0.2)
    img.save("test_in_rnd.png")
    return img


generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=add_noise(image),
    num_inference_steps=5,
    generator=generator,
    guidance_scale=1,
    # guidance_scale=10.5,
    strength=0.8,
    # guidance_scale=0,
).images[0]
image.save("test.png")

from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils import load_image
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"


pipe = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# pipe.load_lora_weights()


pipe.to(device="cuda", dtype=torch.float16)

prompt = "Mesmerizing oil painting of Santa Claus, by ((Henri de Toulouse-Lautrec)), by (Amedeo Modigliani), by (((Tom Thomson)))"
negative_prompt = "text, watermark, frames, border, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
image = load_image("/home/ezalos/42/art/comfyui/swag_mirror/2023-12-31 22:33:11_in.png")
images = pipe(
    prompt=prompt,
	negative_prompt=negative_prompt,
	num_inference_steps=6, 
	guidance_scale=1.6, 
	strength=0.8, 
	image=image
).images[0]


images.save("test.png")

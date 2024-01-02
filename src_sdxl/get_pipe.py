from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    LMSDiscreteScheduler,
    ControlNetModel,
    LatentConsistencyModelPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.utils import load_image
import torch
import PIL
from controlnet_aux import OpenposeDetector

from mirror.get_artist import sample_artist, sample_sentence
from src_depth.model import get_model
from src_depth.preprocess import make_depth_transform, render_depth
from src_depth.inference import infer_depth


def add_noise(x):
    print(f"{x.size = } {x.mode = }")
    # x.save("test_in.png")
    rnd_R = PIL.Image.effect_noise(x.size, sigma=50)
    rnd_G = PIL.Image.effect_noise(x.size, sigma=50)
    rnd_B = PIL.Image.effect_noise(x.size, sigma=50)
    print(f"{rnd_R.size = } {rnd_R.mode = }")
    print(f"{rnd_G.size = } {rnd_G.mode = }")
    print(f"{rnd_B.size = } {rnd_B.mode = }")
    noise = PIL.Image.merge(
        "RGB",
        (
            rnd_R,
            rnd_G,
            rnd_B,
        ),
    )
    print(f"{noise.size = } {noise.mode = }")
    # noise.save("test_rnd.png")
    img = PIL.Image.blend(x, noise, 0.5)
    # img.save("test_in_rnd.png")
    return img


def get_pipe():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

    print("Loading sdxl txt2img pipeline...")
    txt2img_pipe = DiffusionPipeline.from_pretrained(
        model_id,
        # torch_dtype=torch.float16,
        # use_safetensors=True,
        variant="fp16",
    )

    print("Loading Scheduler...")
    txt2img_pipe.load_lora_weights(lcm_lora_id)
    # txt2img_pipe.fuse_lora()
    # print(f"{txt2img_pipe.scheduler.config = }")
    txt2img_pipe.scheduler = LCMScheduler.from_config(txt2img_pipe.scheduler.config)
    # print(f"{txt2img_pipe.scheduler.config = }")
    txt2img_pipe.to(device="cuda", dtype=torch.float16)
    txt2img_pipe.enable_xformers_memory_efficient_attention()
    # txt2img_pipe.unet = torch.compile(
    #     txt2img_pipe.unet, mode="reduce-overhead", fullgraph=True
    # )

    # print("Loading SDXL img2img pipeline...")
    # img2img_pipe = StableDiffusionXLImg2ImgPipeline(
    #     vae=txt2img_pipe.vae,
    #     text_encoder=txt2img_pipe.text_encoder,
    #     text_encoder_2=txt2img_pipe.text_encoder_2,
    #     tokenizer=txt2img_pipe.tokenizer,
    #     tokenizer_2=txt2img_pipe.tokenizer_2,
    #     unet=txt2img_pipe.unet,
    #     scheduler=txt2img_pipe.scheduler,
    # )

    print("Loading controlnets...")
    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    controlnet_openpose = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        # variant="fp16",
        # use_safetensors=True,
        torch_dtype=torch.float16,
    )

    controlnets = [
        controlnet_depth,
        controlnet_openpose,
    ]

    print("Creating final pipeline...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        vae=txt2img_pipe.vae,
        text_encoder=txt2img_pipe.text_encoder,
        text_encoder_2=txt2img_pipe.text_encoder_2,
        tokenizer=txt2img_pipe.tokenizer,
        tokenizer_2=txt2img_pipe.tokenizer_2,
        unet=txt2img_pipe.unet,
        scheduler=txt2img_pipe.scheduler,
        # torch_dtype=torch.float16,
        controlnet=controlnets,
    )

    # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda", dtype=torch.float16)

    prompt = "Mesmerizing oil painting of Santa Claus, by ((Henri de Toulouse-Lautrec)), by (Amedeo Modigliani), by (((Tom Thomson)))"
    negative_prompt = "text, watermark, frames, border, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    depth_transform = make_depth_transform()
    depth_model = get_model()
    # depth_model.enable_xformers_memory_efficient_attention()
    depth_model = torch.compile(depth_model, mode="reduce-overhead", fullgraph=True)
    generator = torch.manual_seed(0)

    # depth_image.save("test_cog_depth.png")
    # openpose_image.save("test_cog_pose.png")
    # control_image = depth_image
    # control_image = openpose_image
    # print(f"{depth_image.size = }")
    # print(f"{openpose_image.size = }")

    def my_pipe(image):
        prompt = (
            f"{sample_sentence()}"
            f"by (({sample_artist()})), "
            f"by ({sample_artist()}), "
            f"by ((({sample_artist()})))"
        )
        print(f"{prompt = }")
        print(f"{image.size = } {image.mode = }")
        image_resized = image.resize((1024, 1024)).convert("RGB")
        print(f"{image_resized.size = } {image_resized.mode = }")
        noisy_image = add_noise(image_resized)
        depth_image = infer_depth(depth_model, depth_transform, image_resized)
        openpose_image = openpose(
            image_resized,
            image_resolution=1024,
            include_body=True,
            include_hand=True,
            include_face=True,
        )
        control_image = [depth_image, openpose_image]
        stylized_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=noisy_image,
            control_image=control_image,
            num_inference_steps=8,
            generator=generator,
            # width=1024,
            # height=1024,
            guidance_scale=1.6,
            # guidance_scale=10.5,
            strength=0.8,
            # guidance_scale=0,
        ).images[0]
        print(f"{stylized_image.size = }")
        return stylized_image

    print(f"Pipeline complete!")
    return my_pipe

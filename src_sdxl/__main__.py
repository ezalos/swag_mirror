from src_sdxl.get_pipe import get_pipe
from diffusers.utils import load_image

if __name__ == "__main__":
    image = load_image(
        "/home/ezalos/42/art/comfyui/swag_mirror/2023-12-31 22:33:11_in.png"
    )
    my_pipe = get_pipe()
    image = my_pipe(image)
    image.save("test_cog.png")

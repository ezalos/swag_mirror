import torch
import urllib
import time

from PIL import Image
from src_depth.model import get_model
from src_depth.preprocess import make_depth_transform, render_depth
from src_depth.inference import infer_depth


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


if __name__ == "__main__":
    EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    image = load_image_from_url(EXAMPLE_IMAGE_URL)
    image.save("img_0.png")

    transform = make_depth_transform()
    model = get_model()

    result = infer_depth(model, transform, image)
    depth_image = render_depth(result.squeeze().cpu())
    depth_image.save("img_0d.png")

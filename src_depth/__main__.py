import torch
import urllib

from PIL import Image
from src_depth.model import get_model


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


image = load_image_from_url(EXAMPLE_IMAGE_URL)
image.save("img_0.png")


import matplotlib
from torchvision import transforms


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ]
    )


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
    colors = colors[:, :, :3]  # Discard alpha component
    return Image.fromarray(colors)


transform = make_depth_transform()

scale_factor = 1
rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
transformed_image = transform(rescaled_image)
batch = transformed_image.unsqueeze(0).cuda()  # Make a batch of one image

model = get_model()

import time

a = time.time()
with torch.inference_mode():
    result = model.whole_inference(batch, img_meta=None, rescale=True)
depth_image = render_depth(result.squeeze().cpu())
print(f"Total time inference: {time.time() - a}")

depth_image.save("img_0d.png")

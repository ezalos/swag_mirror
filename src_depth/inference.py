import time
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image


def infer_depth(model, transform, image: Image):
    scale_factor = 1
    rescaled_image = image.resize(
        (scale_factor * image.width, scale_factor * image.height)
    )
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda()  # Make a batch of one image

    a = time.time()
    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)
    print(f"Total time inference: {time.time() - a}")
    values = result.squeeze().cpu()

    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)
    return to_pil_image(normalized_values)

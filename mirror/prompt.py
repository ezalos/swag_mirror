import json
from mirror.get_artist import sample_artist
from mirror.webcam import get_webcam_img, convert_frame_to_base64

# with open("pipelines/websocket_example.json") as f:
# with open("pipelines/lcm-tupin_api_one.json") as f:
# with open("pipelines/lcm-fast.json") as f:
# with open("pipelines/lcm-depth&pose_api_one.json") as f:


def get_prompt():
    with open("pipelines/ws-lcm.json") as f:
        prompt = json.load(f)

    prompt["6"]["inputs"]["text"] = (
        # f"Mesmerizing oil painting of a person, "
        f"Mesmerizing oil painting of Santa Claus, "
        f"by (({sample_artist()})), "
        f"by ({sample_artist()}), "
        f"by ((({sample_artist()})))"
    )
    return prompt


def add_image_to_prompt(prompt, frame):
    prompt["122"]["inputs"]["image"] = convert_frame_to_base64(frame)
    return prompt

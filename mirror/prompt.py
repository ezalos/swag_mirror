import json
from mirror.get_artist import sample_artist, sample_sentence
from mirror.webcam import get_webcam_img, convert_frame_to_base64


def get_prompt():
    # with open("pipelines/ws-lcm-fast-depth_OP.json") as f:
    with open("pipelines/ws-lcm_out-depth-OP.json") as f:
        prompt = json.load(f)

    prompt["6"]["inputs"]["text"] = (
        # f"Mesmerizing oil painting of a person, "
        f"{sample_sentence()}"
        f"by (({sample_artist()})), "
        f"by ({sample_artist()}), "
        f"by ((({sample_artist()})))"
    )
    return prompt


def add_image_to_prompt(prompt, frame):
    prompt["122"]["inputs"]["image"] = convert_frame_to_base64(frame)
    return prompt


def add_depth_to_prompt(prompt, frame):
    prompt["137"]["inputs"]["image"] = convert_frame_to_base64(frame)
    return prompt


def add_open_pose_to_prompt(prompt, frame):
    prompt["142"]["inputs"]["image"] = convert_frame_to_base64(frame)
    prompt["139"]["inputs"]["strength"] = 1.3
    return prompt

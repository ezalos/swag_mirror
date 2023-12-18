# import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from mirror import config
from mirror.webcam import convert_base64_to_frame
import io
from PIL import Image
import struct
from PIL import Image
import asyncio


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": config.client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        "http://{}/prompt".format(config.server_address), data=data
    )
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "http://{}/view?{}".format(config.server_address, url_values)
    ) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen(
        "http://{}/history/{}".format(config.server_address, prompt_id)
    ) as response:
        return json.loads(response.read())


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]

    print(f"{history = }")
    for o in history["outputs"]:
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            images_output = []
            if "images" in node_output:
                for image in node_output["images"]:
                    image_data = get_image(
                        image["filename"], image["subfolder"], image["type"]
                    )
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def get_frame_from_ws_block(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]

    print(f"{history = }")
    b64_img = history["prompt"][2]["122"]["inputs"]["image"]
    return convert_base64_to_frame(b64_img)


import asyncio
import websockets


async def get_image_from_websocket(websocket):
    # Wait for a binary message (image data)
    msg = await websocket.recv()
    # print(f"{msg = }")

    # Check if the message is binary
    if isinstance(msg, bytes):
        s = struct.calcsize(">II")
        # print(f"{s = }")
        data = memoryview(msg)
        # print(f"{data = }")

        if len(data) > s:
            event, format = struct.unpack_from(">II", data, 0)
            # print(f"{event = }")
            # print(f"{format = }")
            if event == 1 and format == 2:  # 1=PREVIEW_IMAGE, 2=PNG
                image_bytes = data[s:].tobytes()
                image = Image.open(io.BytesIO(image_bytes))
                # print(f"{image = }")
                return image
    return None

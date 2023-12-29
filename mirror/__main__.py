import cv2

# import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import numpy as np
import fire
import time
import asyncio
from mirror import config
from mirror.prompt import get_prompt, add_image_to_prompt
from mirror.websocket import (
    get_images,
    get_frame_from_ws_block,
    get_image_from_websocket,
    queue_prompt,
)
from mirror.webcam import (
    convert_base64_to_frame,
    get_webcam_img,
    convert_frame_to_base64,
    pillow_to_opencv,
    opencv_to_pillow,
)

from PIL import Image
import io

import websockets


async def do():
    # async with websockets.connect(config.uri) as websocket:
    async with websockets.connect(
        config.uri, max_size=10 * 1024 * 1024
    ) as websocket:  # 10 MB limit
        # ws = websocket.WebSocket()
        # ws.connect(config.uri)
        frame_nb = 0
        while True:
            t_0 = time.time()
            frame_nb += 1
            print(f"Getting input {frame_nb = }")
            prompt = get_prompt()
            print("Prompt text ready")
            frame = get_webcam_img()
            print("img ready")
            prompt = add_image_to_prompt(prompt, frame)
            print("Prompt img ready")

            print("Let's queue prompt")
            prompt_id = queue_prompt(prompt)["prompt_id"]
            print("Prompt sent, waiting for response")
            cv2.imshow(f"Future input", frame)
            cv2.waitKey(1)

            print("Let's get result")
            # stylized_frame = get_frame_from_ws_block(ws, prompt)
            # Process image from WebSocket
            # image = asyncio.run(process_image_from_websocket(ws))

            image = None
            while image is None:
                image = await get_image_from_websocket(websocket)
                print(f"{image = }")
            print("We got something !")
            if image:
                pil_image_rgb = image.convert("RGB")
                # Convert to NumPy array
                numpy_image = np.array(pil_image_rgb)
                # Convert from RGB to BGR (OpenCV format)
                open_cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                print("We got it !")
                # v_img = cv2.vconcat([img1, img2])
                if frame.shape[:2] != open_cv_image.shape[:2]:
                    open_cv_image_resized = cv2.resize(
                        open_cv_image, (frame.shape[1], frame.shape[0])
                    )
                else:
                    open_cv_image_resized = open_cv_image

                h_img = cv2.hconcat([frame, open_cv_image_resized])
                cv2.imwrite(f"./output/hort_{frame_nb}.png", h_img)
                # Your code to handle the processed image
                # For example, display the image or save it
                image.save(f"./output/truc_{frame_nb}.png")

                print(f"Showing output {frame_nb = }")
                # cv2.imshow(f"Current input", frame)
                # cv2.waitKey(1)
                cv2.imshow(f"Output", h_img)
                cv2.waitKey(1)
                # pil_image.save(f"./output/{i}_{ii}.png")
            t_1 = time.time()
            print(f"time: {t_1 - t_0}")


from threading import Thread
import queue


async def swag_it(frame):
    async with websockets.connect(
        config.uri, max_size=10 * 1024 * 1024
    ) as websocket:  # 10 MB limit
        prompt = get_prompt()
        prompt = add_image_to_prompt(prompt, frame)
        prompt_id = queue_prompt(prompt)["prompt_id"]
        print("Let's get result")
        # stylized_frame = get_frame_from_ws_block(ws, prompt)
        # Process image from WebSocket
        # image = asyncio.run(process_image_from_websocket(ws))

        image = None
        while image is None:
            image = await get_image_from_websocket(websocket)
            print(f"{image = }")
        print("We got something !")
        if image:
            pil_image_rgb = image.convert("RGB")
            # Convert to NumPy array
            numpy_image = np.array(pil_image_rgb)
            # Convert from RGB to BGR (OpenCV format)
            open_cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            print("We got it !")
            return open_cv_image


def run_async(func, *args):
    """
    Helper function to run an async function in a separate thread
    and get the result via a queue.
    """
    q = queue.Queue()

    def wrapper():
        result = asyncio.run(func(*args))
        q.put(result)

    Thread(target=wrapper).start()
    return q.get()


from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

from flask_cors import CORS  # Import CORS

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes


@app.route("/process", methods=["POST"])
def process_image():
    data = request.json
    image_data = data["image"]

    # Convert base64 to PIL Image
    image_data = image_data.split(",")[1]  # Remove the "data:image/png;base64," part
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    width, height = image.size
    print(f"IN IMG: {width, height = }")

    # Process the image here (this example simply returns the received image)
    frame = pillow_to_opencv(image)
    open_cv_image = run_async(swag_it, frame)
    pil_img = opencv_to_pillow(open_cv_image)
    width, height = pil_img.size
    print(f"OUT IMG: {width, height = }")

    # Convert PIL Image back to base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify(image=f"data:image/png;base64,{img_str}")


def launch(backend=False):
    if backend:
        app.run(debug=True, host="0.0.0.0", port=1111)
    else:
        asyncio.run(do())


if __name__ == "__main__":
    # asyncio.run(do())
    fire.Fire(launch)

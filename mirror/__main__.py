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
)

from PIL import Image
import io

import websockets


async def do():
    # async with websockets.connect(config.uri) as websocket:
    async with websockets.connect(config.uri, max_size=10 * 1024 * 1024) as websocket:  # 10 MB limit
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
                pil_image_rgb = image.convert('RGB')
                # Convert to NumPy array
                numpy_image = np.array(pil_image_rgb)
                # Convert from RGB to BGR (OpenCV format)
                open_cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                print("We got it !")
                # v_img = cv2.vconcat([img1, img2])
                if frame.shape[:2] != open_cv_image.shape[:2]:
                    open_cv_image_resized = cv2.resize(open_cv_image, (frame.shape[1], frame.shape[0]))
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


if __name__ == "__main__":
    asyncio.run(do())

import requests
import cv2
import queue
import threading
import base64
import numpy as np
from PIL import Image
import numpy as np
import cv2
import os


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


vid = VideoCapture(0)


def get_webcam_img(is_live=True) -> str:
    # while is_live:
    frame = vid.read()
    return frame
    # vid.release()


def convert_frame_to_base64(frame):
    # Encode the frame to JPEG format
    # You might choose other formats (like PNG) depending on your needs
    retval, buffer = cv2.imencode(".jpg", frame)
    if not retval:
        raise ValueError("Failed to encode image")

    # Convert the buffer to a byte string
    jpg_as_text = base64.b64encode(buffer)

    # Decode byte string to a regular string
    jpg_as_text = jpg_as_text.decode("utf-8")

    return jpg_as_text


def pillow_to_opencv(pillow_image):
    """
    Convert a Pillow Image to an OpenCV Image.
    :param pillow_image: Image in Pillow format.
    :return: Image in OpenCV format (BGR).
    """
    # Convert to RGB and then to a numpy array.
    opencv_image = np.array(pillow_image.convert("RGB"))
    # Convert RGB to BGR.
    opencv_image = opencv_image[:, :, ::-1].copy()
    return opencv_image


def test_local():
    url = "http://localhost:1111/process"
    encoded_string = convert_frame_to_base64(get_webcam_img())

    # Prepare the JSON payload
    payload = {"image": f"data:image/png;base64,{encoded_string}"}

    # Send POST request to the Flask server
    response = requests.post(url, json=payload)

    # Process the response
    if response.status_code == 200:
        # Decode the received image from base64
        received_data = response.json()["image"]
        received_data = received_data.split(",")[
            1
        ]  # Remove the "data:image/png;base64," part
        img_data = base64.b64decode(received_data)
        with open("received_image.png", "wb") as file:
            file.write(img_data)
        print("Image received and saved as 'received_image.png'")
    else:
        print("Failed to receive response from server")


import shutil


def test_multi_persons():
    url = "http://localhost:1111/process"
    base_path = "data/multi-persons"
    out_path = "data/multi-persons_results"
    shutil.rmtree(out_path, ignore_errors=True)
    os.mkdir(out_path)
    files = list(os.listdir(base_path))
    files.sort()
    for img_name in files:
        pil_img = Image.open(f"{base_path}/{img_name}")
        encoded_string = convert_frame_to_base64(pillow_to_opencv(pil_img))
        # Prepare the JSON payload
        payload = {"image": f"data:image/png;base64,{encoded_string}"}

        # Send POST request to the Flask server
        response = requests.post(url, json=payload)

        # Process the response
        if response.status_code == 200:
            # Decode the received image from base64
            received_data = response.json()["image"]
            # Remove the "data:image/png;base64," part
            received_data = received_data.split(",")[1]
            img_data = base64.b64decode(received_data)
            out_file = f"{out_path}/{img_name}"
            with open(out_file, "wb") as file:
                file.write(img_data)
            print(f"Image received and saved as '{out_file}'")

            received_data = response.json()["open_pose"]
            # Remove the "data:image/png;base64," part
            received_data = received_data.split(",")[1]
            img_data = base64.b64decode(received_data)
            out_file = f"{out_path}/{img_name}.open_pose.png"
            with open(out_file, "wb") as file:
                file.write(img_data)
            print(f"Image received and saved as '{out_file}'")
        else:
            print("Failed to receive response from server")


if __name__ == "__main__":
    test_multi_persons()

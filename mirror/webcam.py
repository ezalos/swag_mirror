import cv2
import queue
import threading
import base64
import numpy as np
from PIL import Image
import numpy as np
import cv2


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


OPENED_CAM = None


def get_webcam_img(is_live=True) -> str:
    if not OPENED_CAM:
        OPENED_CAM = VideoCapture(0)
    # while is_live:
    frame = OPENED_CAM.read()
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


def convert_base64_to_frame(base64_string):
    # Decode the base64 string
    img_data = base64.b64decode(base64_string)

    # Convert to a numpy array
    nparr = np.frombuffer(img_data, np.uint8)

    # Decode image from the numpy array
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image from base64 string")

    return frame


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


def opencv_to_pillow(opencv_image):
    """
    Convert an OpenCV Image to a Pillow Image.
    :param opencv_image: Image in OpenCV format (BGR).
    :return: Image in Pillow format.
    """
    # Convert BGR to RGB.
    pillow_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # Convert to a Pillow Image.
    pillow_image = Image.fromarray(pillow_image)
    return pillow_image

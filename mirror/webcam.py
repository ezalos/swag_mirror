import cv2
import queue
import threading
import base64
import numpy as np


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

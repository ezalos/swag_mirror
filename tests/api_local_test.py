import requests
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


if __name__ == "__main__":
    
	# URL of the Flask server
	url = 'http://localhost:1111/process'  # Change the URL if needed
     
	encoded_string = convert_frame_to_base64(get_webcam_img())
	# Prepare the JSON payload
	payload = {
		'image': f'data:image/png;base64,{encoded_string}'
	}

	# Send POST request to the Flask server
	response = requests.post(url, json=payload)

	# Process the response
	if response.status_code == 200:
		# Decode the received image from base64
		received_data = response.json()['image']
		received_data = received_data.split(',')[1]  # Remove the "data:image/png;base64," part
		img_data = base64.b64decode(received_data)
		with open('received_image.png', 'wb') as file:
			file.write(img_data)
		print("Image received and saved as 'received_image.png'")
	else:
		print("Failed to receive response from server")
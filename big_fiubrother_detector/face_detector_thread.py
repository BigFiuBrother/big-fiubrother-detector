from big_fiubrother_core import StoppableThread
from big_fiubrother_core.messages import SampledFrameMessage, FaceEmbeddingMessage
from big_fiubrother_detector.face_detector_factory import FaceDetectorFactory
import cv2
import numpy as np


class FaceDetectorThread(StoppableThread):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.face_detector = FaceDetectorFactory.build(configuration['face_detector'])

    def _execute(self):
        face_detection_message = self.input_queue.get()

        if face_detection_message is not None:

            # Get message
            frame_id = face_detection_message.camera_id + "_" + face_detection_message.timestamp
            frame_bytes = face_detection_message.frame_bytes

            # Convert to cv2 img
            frame = cv2.imdecode(np.fromstring(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            rects = self.face_detector.detect_face_image(self.frame)
            # check if len(rects) == 0 ?

            # Send message to publisher
            face_embedding_message = FaceEmbeddingMessage(face_detection_message.camera_id,
                                                          face_detection_message.timestamp, frame_id, frame, rects)
            self.output_queue.put(face_embedding_message)

    def _stop(self):
        self.input_queue.put(None)

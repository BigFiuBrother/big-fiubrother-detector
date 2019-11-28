from big_fiubrother_core import QueueTask
from big_fiubrother_core.db import (
    Database,
    Face
)
from big_fiubrother_core.messages import (
    FrameMessage,
    FaceEmbeddingMessage
)
from big_fiubrother_detector.face_detector_factory import FaceDetectorFactory
import cv2
import numpy as np


class FaceDetectionTask(QueueTask):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__(input_queue)
        self.output_queue = output_queue
        self.configuration = configuration

        self.db = None
        self.face_detector = None

    def init(self):
        self.face_detector = FaceDetectorFactory.build(self.configuration['face_detector'])
        self.db = Database(self.configuration['db'])

    def execute_with(self, message):
        face_detection_message: FrameMessage = message

        if face_detection_message is not None:

            # Get message
            video_chunk_id = face_detection_message.video_chunk_id
            frame_id = face_detection_message.frame_id
            frame_bytes = face_detection_message.payload

            # Convert to cv2 img
            frame = cv2.imdecode(np.fromstring(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            rects = self.face_detector.detect_face_image(frame)

            if len(rects) > 0:
                for rect in rects:
                    # Insert detected face to db
                    face = Face(frame_id=frame_id, bounding_box=rect)
                    face_db_id = self.db.add(face)

                    # Get cropped face
                    x1, y1, x2, y2 = rect
                    cropped_face = frame[y1:y2, x1:x2]

                    # Queue face embedding job
                    face_embedding_message = FaceEmbeddingMessage(video_chunk_id, face_db_id, cropped_face)
                    self.output_queue.put(face_embedding_message)
            else:
                # Notify of frame analysis completion
                pass

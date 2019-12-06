from big_fiubrother_core import QueueTask
from big_fiubrother_core.db import (
    Database,
    Face,
    FrameProcess
)
from big_fiubrother_core.messages import (
    FrameMessage,
    FaceEmbeddingMessage,
    ProcessedFrameMessage
)
from big_fiubrother_detector.face_detector_factory import FaceDetectorFactory
import cv2
import numpy as np


class FaceDetectionTask(QueueTask):

    def __init__(self, configuration, input_queue, output_queue_to_classifier, output_queue_to_scheduler):
        super().__init__(input_queue)
        self.output_queue_to_classifier = output_queue_to_classifier
        self.output_queue_to_scheduler = output_queue_to_scheduler
        self.configuration = configuration

        self.db = None
        self.face_detector = None

    def init(self):
        self.face_detector = FaceDetectorFactory.build(self.configuration['face_detector'])
        self.db = Database(self.configuration['db'])

    def close(self):
        self.face_detector.close()
        self.db.close()

    def execute_with(self, message):
        face_detection_message: FrameMessage = message

        # Get message
        video_chunk_id = face_detection_message.video_chunk_id
        frame_id = face_detection_message.frame_id
        frame_bytes = face_detection_message.payload

        # Convert to cv2 img
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        #print("- Performing detection - frame_id: " + str(frame_id))
        #rects = []
        rects = self.face_detector.detect_face_image(frame)

        if len(rects) > 0:
            # Create face process record
            frame_process = FrameProcess(frame_id=frame_id,
                                         total_faces_count=len(rects))
            self.db.add(frame_process)

            for rect in rects:
                x1, y1, x2, y2 = rect
                db_rect = [[x1, y1], [x2, y2]]

                # Insert detected face to db
                face = Face(frame_id=frame_id, bounding_box=db_rect)
                self.db.add(face)

                # Get cropped face
                cropped_face = frame[y1:y2, x1:x2]

                #cv2.imshow("asd", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                #cv2.waitKey(1)

                # Queue face embedding job
                face_embedding_message = FaceEmbeddingMessage(video_chunk_id, face.id, cropped_face)
                self.output_queue_to_classifier.put(face_embedding_message)
        else:
            # Notify scheduler of frame analysis completion
            processed_frame_message = ProcessedFrameMessage(video_chunk_id)
            self.output_queue_to_scheduler.put(processed_frame_message)


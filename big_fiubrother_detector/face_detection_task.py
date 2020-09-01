from big_fiubrother_core import QueueTask
from big_fiubrother_core.db import (
    Database,
    Face
)
from big_fiubrother_core.messages import (
    FrameMessage,
    FaceEmbeddingMessage,
    AnalyzedVideoChunkMessage
)
from big_fiubrother_detector.face_detector_factory import FaceDetectorFactory
import cv2
import numpy as np
from big_fiubrother_core.synchronization import ProcessSynchronizer
import logging

def drawBoxes(im, boxes, color):
    x1 = [i[0] for i in boxes]
    y1 = [i[1] for i in boxes]
    x2 = [i[2] for i in boxes]
    y2 = [i[3] for i in boxes]
    for i in range(len(boxes)):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), color, 1)
    return im


class FaceDetectionTask(QueueTask):

    def __init__(self, configuration, input_queue, output_queue_to_classifier, output_queue_to_scheduler):
        super().__init__(input_queue)
        self.output_queue_to_classifier = output_queue_to_classifier
        self.output_queue_to_scheduler = output_queue_to_scheduler
        self.configuration = configuration

        self.db = None
        self.face_detector = None
        self.process_synchronizer = None

    def init(self):
        self.face_detector = FaceDetectorFactory.build(self.configuration['face_detector'])
        self.db = Database(self.configuration['db'])
        self.process_synchronizer = ProcessSynchronizer(self.configuration['synchronization'])

    def close(self):
        self.face_detector.close()
        self.db.close()
        self.process_synchronizer.close()

    def execute_with(self, message):
        face_detection_message: FrameMessage = message

        # Get message
        video_chunk_id = face_detection_message.video_chunk_id
        frame_id = face_detection_message.frame_id
        frame_bytes = face_detection_message.payload

        # Convert to cv2 img
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Print frame
        cv2.imshow("asd", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # Detect faces
        print("- Performing detection - frame_id: " + str(frame_id) + " - video chunk id: " + str(video_chunk_id))
        logging.debug("- Performing detection - frame_id: " + str(frame_id) + " - video chunk id: " + str(video_chunk_id))

        rects = []
        rects = self.face_detector.detect_face_image(frame)
        #

        if len(rects) > 0:

            faces = []
            for rect in rects:
                x1, y1, x2, y2 = rect
                db_rect = [[x1, y1], [x2, y2]]

                # Insert detected face to db
                face = Face(frame_id=frame_id, bounding_box=db_rect)
                self.db.add(face)
                faces.append(face)

                # Face task sync creation
                self.process_synchronizer.register_face_task(
                    video_chunk_id,
                    frame_id,
                    str(face.id)
                )

            for rect, face in zip(rects, faces):

                # Get cropped face
                x1, y1, x2, y2 = rect
                cropped_face = frame[y1:y2, x1:x2]

                #cv2.imshow("asd", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                #cv2.waitKey(1)

                logging.debug("- Found face, assigned id: " + str(face.id) + " - video chunk id: " + str(video_chunk_id))
                print("- Found face, assigned id: " + str(face.id) + " - video chunk id: " + str(video_chunk_id))

                # Print face with rect
                #cv2.imshow("asd", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

                # Queue face embedding job
                face_embedding_message = FaceEmbeddingMessage(video_chunk_id, face.id, cropped_face)
                self.output_queue_to_classifier.put(face_embedding_message)

        else:

            # Frame task sync completion
            should_notify_scheduler = self.process_synchronizer.complete_frame_task(
                video_chunk_id,
                frame_id
            )

            if should_notify_scheduler:
                print("- Notifying interpolator of video chunk analysis completion, video chunk id: " + str(video_chunk_id))
                logging.debug("- Notifying interpolator of video chunk analysis completion, video chunk id: " + str(video_chunk_id))
                # Notify scheduler of video chunk analysis completion
                scheduler_notification = AnalyzedVideoChunkMessage(video_chunk_id)
                self.output_queue_to_scheduler.put(scheduler_notification)

import sys
import yaml
import cv2
from queue import Queue
from big_fiubrother_core.utils import image_to_bytes
from big_fiubrother_core.messages import FrameMessage
from big_fiubrother_core import (
    PublishToRabbitMQ,
    StoppableThread
)
from big_fiubrother_core.db import (
    Database,
    VideoChunk,
    Frame
)


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("--------------------------------")
        print("This script receives a list of images and sends them to a face detector like a sampler would")
        print("")
        print("Usage: ")
        print("python sampler_mock.py 'image_path1' 'image_path2' ... ")
        print("--------------------------------")

    else:

        print('[*] Configuring sampler-mock')

        # input
        with open("sampler-mock.yml") as config_file:
            configuration = yaml.safe_load(config_file)
        print(configuration)

        # Create database connection
        db = Database(configuration['db'])

        # Create publiser
        to_publisher_queue = Queue()
        publisher_thread = StoppableThread(
                PublishToRabbitMQ(configuration=configuration['publisher'],
                                  input_queue=to_publisher_queue))

        # Simulate Sampler -----------------------------------------------

        print('[*] Configuration finished. Starting sampler-mock!')

        publisher_thread.start()

        # upload video chunk
        video_chunk = VideoChunk(camera_id="camera_1",
                                 timestamp=0.0,
                                 payload="video_chunk".encode())
        db.add(video_chunk)

        # sample frames
        image_paths = sys.argv[1:]
        offset = 0
        for image_path in image_paths:

            image = cv2.imread(image_path)
            frame = Frame(offset=offset,
                          video_chunk_id=video_chunk.id)
            db.add(frame)

            payload = image_to_bytes(image)
            frame_message = FrameMessage(video_chunk_id=video_chunk.id,
                                         frame_id=frame.id,
                                         payload=payload)

            offset += 1

            to_publisher_queue.put(frame_message)

        # ----------------------------------------------------------------

        c = input("[*] Wait a few seconds until all frames are sent then press any key to exit")

        publisher_thread.stop()
        publisher_thread.wait()

        print('[*] sampler-mock stopped!')

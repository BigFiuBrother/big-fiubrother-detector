from queue import Queue
from big_fiubrother_core import setup
from big_fiubrother_core import SignalHandler
from big_fiubrother_detector import FaceDetectorThread, DetectedFaceMessagePublisher, VideoFrameConsumer

if __name__ == "__main__":
    configuration = setup('Big Fiubrother Face Detector Application')

    print('[*] Configuring big_fiubrother_detector')

    consumer_to_detector_queue = Queue()
    detector_to_publisher_queue = Queue()

    detector_thread = FaceDetectorThread(configuration['face_detector'], consumer_to_detector_queue,
                                         detector_to_publisher_queue)
    publisher_thread = DetectedFaceMessagePublisher(configuration['publisher'], detector_to_publisher_queue)
    consumer = VideoFrameConsumer(configuration['consumer'], consumer_to_detector_queue)

    signal_handler = SignalHandler(callback=consumer.stop)

    print('[*] Configuration finished. Starting big_fiubrother_detector!')

    detector_thread.start()
    publisher_thread.start()
    consumer.start()

    # Signal Handled STOP

    detector_thread.stop()
    publisher_thread.stop()

    detector_thread.wait()
    publisher_thread.wait()

    print('[*] big_fiubrother_detector stopped!')
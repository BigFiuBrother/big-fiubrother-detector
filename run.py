from queue import Queue
from big_fiubrother_core import (
    SignalHandler,
    StoppableThread,
    PublishToRabbitMQ,
    ConsumeFromRabbitMQ,
    setup
)
from big_fiubrother_detector import FaceDetectionTask


if __name__ == "__main__":
    configuration = setup('Big Fiubrother Face Detector Application')

    print('[*] Configuring big_fiubrother_detector')

    consumer_to_detector_queue = Queue()
    detector_to_publisher_queue = Queue()

    consumer = ConsumeFromRabbitMQ(configuration=configuration['consumer'],
                                   output_queue=consumer_to_detector_queue)

    detector_thread = StoppableThread(
        FaceDetectionTask(configuration=configuration['face_detector'],
                          input_queue=consumer_to_detector_queue,
                          output_queue=detector_to_publisher_queue))

    publisher_thread = StoppableThread(
        PublishToRabbitMQ(configuration=configuration['publisher'],
                          input_queue=detector_to_publisher_queue))

    signal_handler = SignalHandler(callback=consumer.stop)

    print('[*] Configuration finished. Starting big-fiubrother-detector!')

    detector_thread.start()
    publisher_thread.start()
    consumer.init()
    consumer.execute()

    # Signal Handled STOP
    consumer.close()

    detector_thread.stop()
    publisher_thread.stop()

    detector_thread.wait()
    publisher_thread.wait()

    print('[*] big-fiubrother-detector stopped!')

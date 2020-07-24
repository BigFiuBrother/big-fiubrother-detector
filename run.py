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
    detector_to_classifier_queue = Queue()
    detector_to_scheduler_queue = Queue()

    consumer = ConsumeFromRabbitMQ(configuration=configuration['consumer'],
                                   output_queue=consumer_to_detector_queue)

    detector_thread = StoppableThread(
        FaceDetectionTask(configuration=configuration['face_detector'],
                          input_queue=consumer_to_detector_queue,
                          output_queue_to_classifier=detector_to_classifier_queue,
                          output_queue_to_scheduler=detector_to_scheduler_queue))

    publisher_to_classifier_thread = StoppableThread(
        PublishToRabbitMQ(configuration=configuration['publisher_to_classifier'],
                          input_queue=detector_to_classifier_queue))

    publisher_to_scheduler_thread = StoppableThread(
        PublishToRabbitMQ(configuration=configuration['publisher_to_scheduler'],
                          input_queue=detector_to_scheduler_queue))

    signal_handler = SignalHandler(callback=consumer.stop)

    print('[*] Configuration finished. Starting big-fiubrother-detector!')

    # Start worker threads
    detector_thread.start()
    publisher_to_classifier_thread.start()
    publisher_to_scheduler_thread.start()

    # Start consumer on main thread
    consumer.init()
    consumer.execute()

    # Signal Handled STOP
    consumer.close()

    # Stop worker threads
    detector_thread.stop()
    publisher_to_classifier_thread.stop()
    publisher_to_scheduler_thread.stop()

    # Wait for worker threads
    detector_thread.wait()
    publisher_to_classifier_thread.wait()
    publisher_to_scheduler_thread.wait()

    print('[*] big-fiubrother-detector stopped!')

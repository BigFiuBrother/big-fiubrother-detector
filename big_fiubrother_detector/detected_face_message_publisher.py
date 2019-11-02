from big_fiubrother_core.message_clients.rabbitmq import Publisher
from big_fiubrother_core import StoppableThread


class DetectedFaceMessagePublisher(StoppableThread):

    def __init__(self, publisher_configuration, input_queue):
        super().__init__()
        self.publisher_configuration = publisher_configuration
        self.input_queue = input_queue

    def _init(self):
        self.publisher = Publisher(self.publisher_configuration)

    def _execute(self):
        message = self.input_queue.get()

        if message is not None:
            self.publisher.publish(message)

    def _stop(self):
        self.input_queue.put(None)

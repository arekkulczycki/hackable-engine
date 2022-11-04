# -*- coding: utf-8 -*-
"""
Queue provided by external FasterFifo library.

pip install confluent-kafka==1.9.*
"""

from typing import Tuple, List

from arek_chess.common.queue.base_queue import BaseQueue

from confluent_kafka import Consumer, Producer


class KafkaQueue(BaseQueue):
    """
    Queue provided by external FasterFifo library.
    """

    def __init__(self, name):
        """"""

        super().__init__(name)

        self.consumer = Consumer()
        self.producer = Producer()

        self.consumer.subscribe([name])

    def get_many(self, number_to_get: int = 10) -> List[Tuple]:
        """"""

        consumer = self.consumer
        return [consumer.poll(0) for _ in range(number_to_get)]

    def put_many(self, messages: List[Tuple]) -> None:
        """"""

        name = self.name
        producer = self.producer
        for message in messages:
            producer.produce(name, message)

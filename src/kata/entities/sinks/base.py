import simpy
from kata.entities import Buffer


class Sink:
    id: int
    environment: simpy.Environment
    in_buffer: Buffer

    def _run(self):
        raise NotImplementedError

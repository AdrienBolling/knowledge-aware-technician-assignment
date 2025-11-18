import simpy
from kata.entities import Buffer


class Source:
    env: simpy.Environment
    out_buffer: Buffer

    def _run(self):
        raise NotImplementedError

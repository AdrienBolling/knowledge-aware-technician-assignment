import simpy
from kata.entities import Buffer


class Router:
    env: simpy.Environment
    input_buffer: Buffer

    def _run(self):
        raise NotImplementedError("This is an interface method")

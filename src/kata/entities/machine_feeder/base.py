import simpy
from kata.entities import Buffer


class MachineFeeder:
    env: simpy.Environment
    id: int
    in_buffer: Buffer
    machine_in_buffers: list[Buffer]

    def _run(self):
        raise NotImplementedError

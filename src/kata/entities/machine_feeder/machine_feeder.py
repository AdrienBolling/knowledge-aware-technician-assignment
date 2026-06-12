import logging

import simpy

from kata.entities.buffers.base import Buffer
from kata.entities.machine_feeder.base import MachineFeeder as MachineFeederBase

logger = logging.getLogger(__name__)


class MachineFeeder(MachineFeederBase):
    """A feeder that distributes products from a type-specific buffer
    to multiple machines of that type (load balancing).
    """

    _id_counter = 0

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        machine_type: str,
        in_buffer: Buffer,
        machine_input_buffers: list[Buffer],
        machines: list | None = None,
    ):
        """Initialize a MachineFeeder.

        Args:
            env: SimPy environment
            name: Name of the feeder
            machine_type: Type of machines this feeder serves
            in_buffer: Input buffer (type-specific queue)
            machine_input_buffers: List of input buffers for individual machines
            machines: Optional list of machine instances paired index-wise
                with ``machine_input_buffers``.  When supplied the feeder
                prefers machines whose ``broken`` flag is ``False`` over
                broken ones, so products do not pile up behind a machine
                that is waiting on repair.  Falls back to least-loaded
                selection across all machines when every machine is broken.

        """
        self.env = env
        self.id = MachineFeeder._id_counter
        MachineFeeder._id_counter += 1
        self.name = name
        self.machine_type = machine_type
        self.in_buffer = in_buffer
        self.machine_input_buffers = machine_input_buffers
        self.machines = machines

        self.fed = 0
        self.current_machine_idx = 0  # Round-robin index
        self.proc = env.process(self._run())

    def _log(self, *args) -> None:
        """Log a message with timestamp and feeder name."""
        logger.debug(
            "[%8.1f] [FEEDER:%s] %s",
            self.env.now,
            self.name,
            " ".join(str(a) for a in args),
        )

    def _run(self):
        """Feed each product to the working machine with the shortest input queue.

        Selection order:
        1. Among working (non-broken) machines, pick the one with the
           fewest queued items in its input buffer.
        2. If every machine is broken (or no ``machines`` list was passed),
           fall back to the least-loaded buffer across all machines.

        Round-robin causes the feeder to block as soon as one machine's
        input buffer fills up — typically because that machine is broken
        and waiting for repair — which back-pressures the type-queue,
        blocks the router, and stalls the whole factory.  Skipping broken
        machines whenever a working one is available keeps WIP from
        piling up behind a machine that cannot drain it.
        """
        while True:
            product = yield self.in_buffer.get()

            indices = range(len(self.machine_input_buffers))
            if self.machines is not None:
                working = [i for i in indices if not self.machines[i].broken]
                candidates = working or list(indices)
            else:
                candidates = list(indices)

            target_idx = min(
                candidates,
                key=lambda i: len(self.machine_input_buffers[i].store.items),
            )
            target_buffer = self.machine_input_buffers[target_idx]
            self._log(f"Feeding product {product.product_id} to {target_buffer.name}")
            yield target_buffer.put(product)
            self.fed += 1

            # ``current_machine_idx`` is kept for backwards compatibility with
            # any consumer that inspects it for telemetry; it now just trails
            # the most recent target.
            self.current_machine_idx = target_idx

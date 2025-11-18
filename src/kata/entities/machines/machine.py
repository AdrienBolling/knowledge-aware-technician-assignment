import simpy as sp

from kata.entities.products.product import Product
from kata.entities.technicians.technician import Technician, TechDispatcher
from kata.features.breakdown.base import BreakdownProcess


class Machine:
    """
    A machine works on products abd may get broken
    When it breaks, creates a ticket for repair and requetss a Technician.
    Resumes normal production after it hasbeen repaired.
    """

    def __init__(
        self,
        env: sp.Environment,
        machine_id: int,
        mtype: str,
        input_buffer: sp.Store,
        output_buffer: sp.Store,
        tech_dispatcher: TechDispatcher,
        breakdown_process: BreakdownProcess,
        process_time: int,
        dt: int,
    ) -> None:
        self.env = env
        self.machine_id = machine_id
        self.mtype = mtype
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.tech_dispatcher = tech_dispatcher
        self.process_time = process_time
        self.breakdown_process = breakdown_process

        self.broken = False
        self.total_processed = 0
        self.last_failed_at = None

        self.proc = env.process(self._run())
        self.breaks = env.process(self._breakdown_driver())

        self.is_processing = False

        self.dt = dt

    def _log(self, *args) -> None:
        print(f"[{self.env.now:05}] - [M: {self.machine_id}] ", *args)

    def _run(self):
        while True:
            # Check if the machine is broken, if so wait for a repair
            if self.broken:
                self._log("is broken, waiting for repair")
                yield self.tech_dispatcher.wait_until_repaired(self)

            ## Process Product
            # Pull product from input buffer
            product = yield self.input_buffer.get()
            ptime = self.process_time
            start = self.env.now
            remaining = float(ptime)
            self._log(f"starts processing product {product.product_id} for {ptime:.2f}")
            while remaining > 0.0:
                try:
                    self.is_processing = True
                    yield self.env.timeout(delay=remaining)
                    remaining: int = 0
                except sp.Interrupt as e:
                    if not self.broken:
                        continue
                    remaining = max(
                        0.0, float(start) + float(ptime) - float(self.env.now)
                    )
                    self._log(
                        f"interrupted due to breakdown, waiting for repair. Remaining time: {remaining}"
                    )
                    yield self.tech_dispatcher.wait_until_repaired(self)
                    self._log("restarting processing of product", product.product_id)

            # Try to move to the output buffer (may block if full)
            self.is_processing = False
            self._log(
                f"finished processing product {product.product_id}, enqueue to {self.output_buffer.name}"
            )
            yield self.output_buffer.put(product)
            self.total_processed += 1
            self._log(
                f"product {product.product_id} enqueued succesfully to {self.output_buffer}"
            )

    def _breakdown_driver(self):
        while True:
            yield self.env.timeout(self.dt)
            if self.broken:
                continue
            p_break = (
                self.breakdown_process.step_and_get_proba()
                if self.is_processing
                else self.breakdown_process.step_and_get_idle_proba()
            )
            if p_break >= self.env.random.uniform(0, 1) and p_break > 0:
                # Machine breaks down
                self._trigger_breakdown()

    def _trigger_breakdown(self):
        self.broken = True
        self.last_failed_at = self.env.now
        self._log("BREAKDOWN occurred!")
        if self.proc.is_alive:
            try:
                self.proc.interrupt("breakdown")
            except RuntimeError:
                pass
            self.tech_dispatcher.request_repair(self)

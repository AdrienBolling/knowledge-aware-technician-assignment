"""
SimPy OOP production line with breakdowns + technician assignment hook for RL.

What you get:
- Product flows with routing (list of machine types)
- Machines with downstream buffers (blocking on full buffer)
- Stochastic processing, failures, repair durations
- Technician pool; agent decides which technician is dispatched on each breakdown
- Clean hook (AgentPolicy) where you can plug an RL policy

Requires: simpy, numpy

Run: python this_file.py
"""

from __future__ import annotations
import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, dict, Optional, Any

# ---------------------------- Utility RNG wrappers ---------------------------


class Dist:
    """Callable distribution factory for clarity."""

    def __init__(
        self, sampler: Callable[[np.random.Generator], float], name: str = "dist"
    ):
        self.sampler = sampler
        self.name = name

    def __call__(self, rng: np.random.Generator) -> float:
        return float(self.sampler(rng))

    @staticmethod
    def constant(v: float) -> "Dist":
        return Dist(lambda rng: v, name=f"const({v})")

    @staticmethod
    def exp(mean: float) -> "Dist":
        lam = 1.0 / mean
        return Dist(lambda rng: rng.exponential(1 / lam), name=f"exp(mean={mean})")

    @staticmethod
    def normal(mean: float, sd: float, min_clip: float = 0.0) -> "Dist":
        def f(rng):
            return max(min_clip, rng.normal(mean, sd))

        return Dist(f, name=f"norm({mean},{sd})")


# --------------------------------- Domain -----------------------------------


@dataclass
class Product:
    pid: int
    route: List[str]  # list of machine_type names
    step: int = 0

    def next_machine_type(self) -> Optional[str]:
        return self.route[self.step] if self.step < len(self.route) else None

    def advance(self):
        self.step += 1


@dataclass
class MachineType:
    name: str
    proc_time: Dist
    mtbf: Dist  # mean time between failures distribution (time to fail)
    mttr: Dist  # mean time to repair distribution (base repair duration)


class Buffer:
    def __init__(self, env: simpy.Environment, name: str, capacity: int):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=capacity)

    def put(self, item):
        return self.store.put(item)

    def get(self):
        return self.store.get()

    def __len__(self):
        return len(self.store.items)


# ------------------------ Technician & Repair Requests -----------------------


@dataclass
class Technician:
    tid: int
    name: str
    travel_time: Dist
    efficiency: float = (
        1.0  # multiplies base repair time: effective = base / efficiency
    )

    def __post_init__(self):
        self.busy: bool = False


@dataclass
class RepairRequest:
    machine: "Machine"
    created_at: float
    chosen_tid: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------- Agent Interface ------------------------------


class AgentPolicy:
    """Plug your RL agent here. Provide state via request/meta and tech list."""

    def select_technician(
        self, request: RepairRequest, technicians: List[Technician]
    ) -> int:
        raise NotImplementedError


class RandomAgent(AgentPolicy):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_technician(
        self, request: RepairRequest, technicians: List[Technician]
    ) -> int:
        free = [t for t in technicians if not t.busy]
        if not free:
            # If everyone is busy, pick the one that becomes available first would be better.
            # For this simple example, just choose uniformly among all.
            return int(self.rng.integers(0, len(technicians)))
        idxs = [tech.tid for tech in free]
        return int(self.rng.choice(idxs))


# --------------------------------- Machine ----------------------------------


class Machine:
    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        mtype: MachineType,
        input_buffer: Buffer,
        output_buffer: Buffer,
        tech_dispatcher: "TechDispatcher",
        rng: np.random.Generator,
    ):
        self.env = env
        self.name = name
        self.type = mtype
        self.in_buf = input_buffer
        self.out_buf = output_buffer
        self.dispatcher = tech_dispatcher
        self.rng = rng

        self.broken = False
        self.total_processed = 0
        self.last_failed_at: Optional[float] = None

        self.proc = env.process(self._run())
        self.breaks = env.process(self._breakdown_driver())

    def _log(self, *args):
        print(f"[{self.env.now:8.3f}] [M:{self.name}]", *args)

    def _run(self):
        while True:
            # Pull product; this blocks if no input
            product: Product = yield self.in_buf.get()
            # If broken, wait until repaired (product stays on machine)
            if self.broken:
                self._log(
                    "is broken; waiting for repair before processing product",
                    product.pid,
                )
                # wait for repair completion event
                yield self.dispatcher.wait_until_repaired(self)

            # Process product
            ptime = self.type.proc_time(self.rng)
            self._log(f"start processing product {product.pid} for {ptime:.2f}")
            try:
                yield self.env.timeout(ptime)
            except simpy.Interrupt:
                # If interrupted by a failure, we will block until repaired and then resume remaining time.
                # For simplicity, we restart full process time after repair (alternative: track remaining time).
                self._log("interrupted by breakdown; waiting repair to restart")
                yield self.dispatcher.wait_until_repaired(self)
                self._log("restarting processing after repair")
                yield self.env.timeout(ptime)

            # Try to move into output buffer (may block if full)
            self._log(f"finished product {product.pid}; enqueue to {self.out_buf.name}")
            yield self.out_buf.put(product)
            self.total_processed += 1
            self._log(f"moved product {product.pid} to {self.out_buf.name}")

    def _breakdown_driver(self):
        while True:
            ttf = self.type.mtbf(self.rng)
            yield self.env.timeout(ttf)
            # breakdown occurs
            if self.broken:
                continue  # already broken (shouldn't happen in this simple model)
            self.broken = True
            self.last_failed_at = self.env.now
            self._log("*** BREAKDOWN occurred ***")
            # Ask dispatcher (which calls the agent) to send a technician
            self.dispatcher.request_repair(self)
            # Block until repaired
            yield self.dispatcher.wait_until_repaired(self)
            self._log("+++ repaired; resume operations +++")


# ------------------------ Technician Dispatch & Repairs ----------------------


class TechDispatcher:
    def __init__(
        self,
        env: simpy.Environment,
        technicians: List[Technician],
        agent: AgentPolicy,
        rng: np.random.Generator,
    ):
        self.env = env
        self.techs = technicians
        self.agent = agent
        self.rng = rng
        # map machine -> repair completion event
        self._repair_events: dict[Machine, simpy.Event] = {}
        # simpy.Resource for each tech to model busy/idle
        self._tech_resources: dict[int, simpy.Resource] = {
            t.tid: simpy.Resource(env, capacity=1) for t in self.techs
        }

    # External API used by Machine
    def request_repair(self, machine: Machine):
        # Create request and let the policy pick a tech
        req = RepairRequest(
            machine=machine,
            created_at=self.env.now,
            meta={
                "machine_type": machine.type.name,
                "queue_len": len(machine.in_buf),
            },
        )
        chosen_tid = self.agent.select_technician(req, self.techs)
        req.chosen_tid = chosen_tid
        tech = self._get_tech(chosen_tid)
        # Start a repair job process
        self.env.process(self._repair_job(tech, machine))

    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        if machine not in self._repair_events:
            self._repair_events[machine] = self.env.event()
        return self._repair_events[machine]

    # Internal helpers
    def _get_tech(self, tid: int) -> Technician:
        for t in self.techs:
            if t.tid == tid:
                return t
        raise KeyError(f"Technician {tid} not found")

    def _repair_job(self, tech: Technician, machine: Machine):
        res = self._tech_resources[tech.tid]
        machine._log(f"tech request -> {tech.name}")
        with res.request() as req:
            yield req  # wait for tech to be free
            tech.busy = True
            # travel time
            t_travel = tech.travel_time(self.rng)
            machine._log(f"{tech.name} traveling for {t_travel:.2f}")
            yield self.env.timeout(t_travel)
            # repair time
            base = machine.type.mttr(self.rng)
            t_rep = base / max(1e-6, tech.efficiency)
            machine._log(
                f"{tech.name} repairing for {t_rep:.2f} (base {base:.2f} / eff {tech.efficiency:.2f})"
            )
            yield self.env.timeout(t_rep)
            # mark repaired
            machine.broken = False
            if machine in self._repair_events:
                self._repair_events[machine].succeed()
                del self._repair_events[machine]
            machine._log(f"{tech.name} repair complete")
            tech.busy = False


# ------------------------------ Factory Layout -------------------------------


class ProductionLine:
    """Linear(ish) layout where machines are grouped by type; products follow routes by type.
    You can easily extend to a graph. This example wires a simple two-stage flow: SRC -> M* -> SINK.
    """

    def __init__(
        self,
        env: simpy.Environment,
        tech_dispatcher: TechDispatcher,
        rng: np.random.Generator,
    ):
        self.env = env
        self.dispatcher = tech_dispatcher
        self.rng = rng
        self.buffers: dict[str, Buffer] = {}
        self.machines: List[Machine] = []
        self.machine_types: dict[str, MachineType] = {}

    def add_machine_type(self, mtype: MachineType):
        self.machine_types[mtype.name] = mtype

    def add_buffer(self, name: str, capacity: int) -> Buffer:
        buf = Buffer(self.env, name, capacity)
        self.buffers[name] = buf
        return buf

    def add_machine(
        self, name: str, type_name: str, input_buffer: Buffer, output_buffer: Buffer
    ) -> Machine:
        mtype = self.machine_types[type_name]
        m = Machine(
            self.env,
            name,
            mtype,
            input_buffer,
            output_buffer,
            self.dispatcher,
            self.rng,
        )
        self.machines.append(m)
        return m


# ----------------------------- Source / Sink --------------------------------


class Source:
    def __init__(
        self,
        env: simpy.Environment,
        out_buf: Buffer,
        interarrival: Dist,
        route: List[str],
        rng: np.random.Generator,
        max_products: Optional[int] = None,
    ):
        self.env = env
        self.out = out_buf
        self.interarrival = interarrival
        self.route = route
        self.rng = rng
        self.max_products = max_products
        self.pid_counter = 0
        self.proc = env.process(self._run())

    def _run(self):
        while self.max_products is None or self.pid_counter < self.max_products:
            ia = self.interarrival(self.rng)
            yield self.env.timeout(ia)
            pid = self.pid_counter
            self.pid_counter += 1
            p = Product(pid=pid, route=list(self.route))
            print(f"[{self.env.now:8.3f}] [SRC] new product {p.pid} -> {self.out.name}")
            yield self.out.put(p)


class Sink:
    def __init__(self, env: simpy.Environment, in_buf: Buffer):
        self.env = env
        self.in_buf = in_buf
        self.completed: int = 0
        self.proc = env.process(self._run())

    def _run(self):
        while True:
            p: Product = yield self.in_buf.get()
            self.completed += 1
            print(
                f"[{self.env.now:8.3f}] [SNK] product {p.pid} completed (total={self.completed})"
            )


# --------------------------- Router between stages ---------------------------


class Router:
    """Consumes from a buffer and routes product to the correct next-stage buffer based on route."""

    def __init__(
        self, env: simpy.Environment, in_buf: Buffer, type_to_buffer: dict[str, Buffer]
    ):
        self.env = env
        self.in_buf = in_buf
        self.type_to_buffer = type_to_buffer
        self.proc = env.process(self._run())

    def _run(self):
        while True:
            p: Product = yield self.in_buf.get()
            # determine next machine type
            mtype = p.next_machine_type()
            if mtype is None:
                # No next type: send to a special buffer named "SINK" if provided
                buf = self.type_to_buffer.get("__SINK__")
                if buf is None:
                    raise RuntimeError("No sink buffer provided for finished product")
                yield buf.put(p)
                continue
            # Otherwise forward to the buffer for that type and advance step when machine pulls it
            outb = self.type_to_buffer[mtype]
            yield outb.put(p)


# ----------------------- Machine input wrapper (router pop) ------------------


class MachineFeeder:
    """Pulls products destined for this machine type, advances route, and feeds machines' input buffers.
    Use one feeder per machine type, connected to the router's output buffer for that type.
    """

    def __init__(
        self,
        env: simpy.Environment,
        type_name: str,
        in_buf: Buffer,
        machine_inputs: List[Buffer],
    ):
        self.env = env
        self.type_name = type_name
        self.in_buf = in_buf
        self.machine_inputs = machine_inputs
        self.idx = 0  # round-robin assignment across same-type machines
        self.proc = env.process(self._run())

    def _run(self):
        while True:
            p: Product = yield self.in_buf.get()
            # Advance because we're entering a machine of this type next
            p.advance()
            # Round-robin to machines of this type
            target = self.machine_inputs[self.idx % len(self.machine_inputs)]
            self.idx += 1
            yield target.put(p)


# ---------------------------------- Demo ------------------------------------


def build_demo_env(seed: int = 0):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    # Agent & technicians
    techs = [
        Technician(
            tid=0, name="T_A", travel_time=Dist.normal(3.0, 1.0, 0.5), efficiency=1.0
        ),
        Technician(
            tid=1, name="T_B", travel_time=Dist.normal(1.0, 0.3, 0.2), efficiency=0.7
        ),
        Technician(
            tid=2, name="T_C", travel_time=Dist.normal(5.0, 1.5, 0.5), efficiency=1.3
        ),
    ]
    agent = RandomAgent(rng)
    dispatcher = TechDispatcher(env, technicians=techs, agent=agent, rng=rng)

    # Production line
    line = ProductionLine(env, dispatcher, rng)

    # Define machine types
    line.add_machine_type(
        MachineType(
            name="Drill",
            proc_time=Dist.normal(5.0, 1.0, 0.5),
            mtbf=Dist.exp(60.0),
            mttr=Dist.normal(10.0, 3.0, 2.0),
        )
    )
    line.add_machine_type(
        MachineType(
            name="Paint",
            proc_time=Dist.normal(8.0, 2.0, 1.0),
            mtbf=Dist.exp(90.0),
            mttr=Dist.normal(12.0, 4.0, 3.0),
        )
    )

    # Buffers
    src_out = line.add_buffer("BUF_SRC", capacity=50)
    route_out = line.add_buffer("BUF_ROUTE", capacity=100)

    # Per-type buffers (from router to feeder) and machine I/O
    drill_q = line.add_buffer("BUF_DRILL_IN", capacity=30)
    paint_q = line.add_buffer("BUF_PAINT_IN", capacity=30)

    # Machine inputs (fed by feeders) and outputs
    m1_in = line.add_buffer("BUF_M1_IN", capacity=5)
    m1_out = line.add_buffer("BUF_M1_OUT", capacity=5)
    m2_in = line.add_buffer("BUF_M2_IN", capacity=5)
    m2_out = line.add_buffer("BUF_M2_OUT", capacity=5)

    p1_in = line.add_buffer("BUF_P1_IN", capacity=5)
    p1_out = line.add_buffer("BUF_P1_OUT", capacity=5)

    sink_in = line.add_buffer("BUF_SINK", capacity=100)

    # Machines (2 Drills, 1 Paint) wired in a general router pattern
    m1 = line.add_machine("Drill#1", "Drill", m1_in, m1_out)
    m2 = line.add_machine("Drill#2", "Drill", m2_in, m2_out)
    p1 = line.add_machine("Paint#1", "Paint", p1_in, p1_out)

    # Router: source -> route_out, then Router sends to per-type queues, then feeders fan-in to machine inputs
    type_to_buffer = {"Drill": drill_q, "Paint": paint_q, "__SINK__": sink_in}
    router = Router(env, route_out, type_to_buffer)

    # Feeders
    drill_feeder = MachineFeeder(env, "Drill", drill_q, [m1_in, m2_in])
    paint_feeder = MachineFeeder(env, "Paint", paint_q, [p1_in])

    # Wire: source emits to route_out
    source = Source(
        env,
        out_buf=route_out,
        interarrival=Dist.exp(7.0),
        route=["Drill", "Paint"],
        rng=rng,
        max_products=50,
    )

    # Wire machine stage outputs back to router input for next step
    # After Drill, product needs Paint => push M*_OUT to route_out
    env.process(_conveyor(env, m1_out, route_out, name="M1->Route"))
    env.process(_conveyor(env, m2_out, route_out, name="M2->Route"))

    # After Paint, finished => push to sink
    env.process(_conveyor(env, p1_out, sink_in, name="P1->Sink"))

    # Sink
    sink = Sink(env, sink_in)

    return env


def _conveyor(
    env: simpy.Environment,
    src: Buffer,
    dst: Buffer,
    name: str = "conv",
    delay: float = 0.0,
):
    while True:
        item = yield src.get()
        if delay > 0:
            yield env.timeout(delay)
        print(f"[{env.now:8.3f}] [CNV:{name}] move product {item.pid} -> {dst.name}")
        yield dst.put(item)


# --------------------------------- Metrics ----------------------------------


class KPILogger:
    def __init__(self, env: simpy.Environment, line: ProductionLine, sink: Sink):
        self.env = env
        self.line = line
        self.sink = sink
        self.proc = env.process(self._run())

    def _run(self):
        while True:
            yield self.env.timeout(30.0)
            wip = sum(len(b) for b in self.line.buffers.values())
            done = self.sink.completed
            print(f"[{self.env.now:8.3f}] [KPI] WIP={wip:3d}  DONE={done:3d}")


# ---------------------------------- Main ------------------------------------


def main():
    env = build_demo_env(seed=42)
    # Run for a simulated horizon
    env.run(until=400)


if __name__ == "__main__":
    main()


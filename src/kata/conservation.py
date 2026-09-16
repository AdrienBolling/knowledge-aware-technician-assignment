"""Product conservation count for a factory built by ``ScenarioBuilder``.

Every product that a source creates is, at any instant, in exactly one of
these places:

* finished: consumed by a sink;
* in a buffer: in the ``items`` of a store, or held by a pending put
  request in its ``put_queue`` (a producer that is blocked on a full store);
* in a machine: taken from the input buffer and not yet offered to the
  output buffer (``Machine.current_product``), or held by a feeder that
  waits for a machine of its type (``MachineFeeder.holding``);
* in flight: a store already handed it to a get request whose process has
  not resumed yet (the event is scheduled at the current instant);
* scrapped: drained from the input buffer of a retired machine (lifecycle
  scenarios, deliberate).

``products_lost = created - finished - wip - scrapped`` is therefore zero
unless the simulation destroys products.  The pre-fix ``Machine._run``
(``KATA_LEGACY_BUFFER_INTERRUPT=1``) does: a get request that a breakdown
interrupts stays queued and swallows the next product.
"""

from __future__ import annotations

from typing import Any

from simpy.resources.store import StoreGet


def _store(buffer: Any) -> Any:
    return getattr(buffer, "store", buffer)


def _buffer_products(buffer: Any) -> int:
    """Products in a store: its items plus the items of pending put requests."""
    store = _store(buffer)
    items = len(getattr(store, "items", ()))
    pending = sum(
        1 for ev in getattr(store, "put_queue", ()) if not ev.triggered
    )
    return items + pending


def _in_flight_products(sim_env: Any) -> int:
    """Products handed to a get request whose process has not resumed yet.

    A stale request (its process moved on after an interrupt) is not
    counted: nothing will ever read its value.
    """
    n = 0
    for entry in getattr(sim_env, "_queue", ()):
        ev = entry[-1]
        if not isinstance(ev, StoreGet) or not ev.triggered:
            continue
        if ev.callbacks is None:  # already processed
            continue
        proc = getattr(ev, "proc", None)
        if proc is not None and proc.is_alive and proc.target is ev:
            n += 1
    return n


def product_conservation(sim_env: Any, dispatcher: Any) -> dict[str, int]:
    """Return the product-conservation counts of a live factory.

    Needs the ``factory_handles`` that ``ScenarioBuilder.build`` attaches to
    the dispatcher.  Without them every count is ``-1``.
    """
    keys = (
        "products_created",
        "products_finished",
        "products_wip_buffers",
        "products_wip_machines",
        "products_in_flight",
        "products_wip",
        "products_scrapped",
        "products_lost",
    )
    handles = getattr(dispatcher, "factory_handles", None)
    if handles is None or not getattr(handles, "sources", None):
        return dict.fromkeys(keys, -1)

    created = sum(int(getattr(s, "products_created", 0)) for s in handles.sources)
    finished = sum(
        int(getattr(s, "completed", 0)) for s in (getattr(dispatcher, "sinks", None) or [])
    )

    buffers: dict[int, Any] = {}

    def _add(buf: Any) -> None:
        if buf is not None:
            buffers.setdefault(id(_store(buf)), buf)

    _add(handles.route_buffer)
    _add(handles.sink_buffer)
    for buf in handles.type_queues.values():
        _add(buf)
    machines = list(handles.all_machines)
    for m in machines:
        _add(getattr(m, "input_buffer", None))
        _add(getattr(m, "output_buffer", None))
    in_buffers = sum(_buffer_products(b) for b in buffers.values())

    in_machines = sum(
        1 for m in machines if getattr(m, "current_product", None) is not None
    )
    in_machines += sum(
        1
        for f in handles.feeders.values()
        if getattr(f, "holding", None) is not None
    )
    in_flight = _in_flight_products(sim_env)
    wip = in_buffers + in_machines + in_flight
    scrapped = int(getattr(handles, "products_scrapped", 0))
    return {
        "products_created": created,
        "products_finished": finished,
        "products_wip_buffers": in_buffers,
        "products_wip_machines": in_machines,
        "products_in_flight": in_flight,
        "products_wip": wip,
        "products_scrapped": scrapped,
        "products_lost": created - finished - wip - scrapped,
    }

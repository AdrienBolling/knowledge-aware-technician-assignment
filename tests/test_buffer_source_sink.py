"""Tests for Buffer, Source, Sink, Router, and MachineFeeder entities."""

import simpy

from kata.entities.buffers.buffer import Buffer
from kata.entities.machine_feeder.machine_feeder import MachineFeeder
from kata.entities.products.product import Product
from kata.entities.routers.router import Router
from kata.entities.sinks.sink import Sink
from kata.entities.sources.source import Source


class TestBuffer:
    def test_put_and_get(self):
        env = simpy.Environment()
        buf = Buffer(env, "test_buf", capacity=10)

        def _process():
            yield buf.put("item_a")
            item = yield buf.get()
            assert item == "item_a"

        env.process(_process())
        env.run()

    def test_len(self):
        env = simpy.Environment()
        buf = Buffer(env, "test_buf")
        buf.store.items.extend(["a", "b", "c"])
        assert len(buf) == 3

    def test_items_property(self):
        env = simpy.Environment()
        buf = Buffer(env, "test_buf")
        buf.store.items.extend(["x", "y"])
        assert buf.items == ["x", "y"]

    def test_capacity_is_set(self):
        env = simpy.Environment()
        buf = Buffer(env, "test_buf", capacity=5)
        assert buf.capacity == 5


class TestSource:
    def test_creates_products_with_route(self):
        env = simpy.Environment()
        buf = Buffer(env, "out_buf")
        source = Source(
            env,
            "test_source",
            buf,
            interarrival_time=5.0,
            route=["A", "B"],
            max_products=3,
        )

        env.run(until=20)

        assert len(buf) == 3
        product = buf.items[0]
        assert isinstance(product, Product)
        assert product.route == ["A", "B"]

    def test_respects_max_products(self):
        env = simpy.Environment()
        buf = Buffer(env, "out_buf")
        Source(env, "test_source", buf, interarrival_time=1.0, max_products=2)
        env.run(until=100)
        assert len(buf) == 2


class TestSink:
    def test_consumes_products(self):
        env = simpy.Environment()
        buf = Buffer(env, "sink_buf")
        sink = Sink(env, "test_sink", buf)

        def _feeder():
            for i in range(3):
                yield buf.put(Product(i, []))

        env.process(_feeder())
        env.run(until=10)

        assert sink.completed == 3


class TestRouter:
    def test_routes_to_correct_type_buffer(self):
        env = simpy.Environment()
        in_buf = Buffer(env, "in")
        drill_buf = Buffer(env, "drill")
        paint_buf = Buffer(env, "paint")

        Router(
            env,
            "router",
            in_buf,
            {"Drill": drill_buf, "Paint": paint_buf},
        )

        product = Product(0, ["Drill", "Paint"])
        in_buf.store.items.append(product)

        env.run(until=5)
        assert len(drill_buf) == 1

    def test_routes_completed_product_to_sink(self):
        env = simpy.Environment()
        in_buf = Buffer(env, "in")
        sink_buf = Buffer(env, "sink")

        Router(env, "router", in_buf, {"__SINK__": sink_buf})

        product = Product(0, [])  # already complete
        in_buf.store.items.append(product)

        env.run(until=5)
        assert len(sink_buf) == 1


class TestMachineFeeder:
    def test_round_robin_distribution(self):
        env = simpy.Environment()
        in_buf = Buffer(env, "type_q")
        m1_buf = Buffer(env, "m1_in")
        m2_buf = Buffer(env, "m2_in")

        MachineFeeder(env, "feeder", "test", in_buf, [m1_buf, m2_buf])

        # Add 4 products
        for i in range(4):
            in_buf.store.items.append(Product(i, ["test"]))

        env.run(until=10)

        assert len(m1_buf) == 2
        assert len(m2_buf) == 2

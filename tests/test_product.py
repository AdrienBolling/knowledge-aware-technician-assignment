"""Tests for the Product entity."""

from kata.entities.products.product import Product


class TestProduct:
    def test_creation(self):
        p = Product(product_id=0, route=["A", "B", "C"])
        assert p.product_id == 0
        assert p.route == ["A", "B", "C"]
        assert p.step == 0

    def test_next_machine_type(self):
        p = Product(0, ["A", "B"])
        assert p.next_machine_type() == "A"

    def test_advance(self):
        p = Product(0, ["A", "B"])
        p.advance()
        assert p.step == 1
        assert p.next_machine_type() == "B"

    def test_is_complete(self):
        p = Product(0, ["A"])
        assert not p.is_complete()
        p.advance()
        assert p.is_complete()

    def test_next_machine_type_returns_none_when_complete(self):
        p = Product(0, ["A"])
        p.advance()
        assert p.next_machine_type() is None

    def test_empty_route_is_immediately_complete(self):
        p = Product(0, [])
        assert p.is_complete()
        assert p.next_machine_type() is None

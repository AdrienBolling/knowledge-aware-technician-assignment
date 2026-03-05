import simpy

from kata.entities.sinks.base import Sink as SinkBase
from kata.entities.buffers.base import Buffer
from kata.entities.products.product import Product


class Sink(SinkBase):
    """A sink that consumes completed products from an input buffer."""
    _id_counter = 0

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        in_buffer: Buffer,
    ):
        """
        Initialize a Sink.
        
        Args:
            env: SimPy environment
            name: Name of the sink
            in_buffer: Input buffer to receive products from
        """
        self.env = env
        self.id = Sink._id_counter
        Sink._id_counter += 1
        self.name = name
        self.in_buffer = in_buffer
        
        self.completed = 0
        self.proc = env.process(self._run())
    
    def _log(self, *args) -> None:
        """Log a message with timestamp and sink name."""
        print(f"[{self.env.now:8.1f}] [SINK:{self.name}]", *args)
    
    def _run(self):
        """Generator process that consumes products."""
        while True:
            # Get product from input buffer
            product: Product = yield self.in_buffer.get()
            self.completed += 1
            self._log(f"Received completed product {product.product_id} (total: {self.completed})")

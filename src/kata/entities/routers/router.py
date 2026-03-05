import simpy

from kata.entities.routers.base import Router as RouterBase
from kata.entities.buffers.base import Buffer
from kata.entities.products.product import Product


class Router(RouterBase):
    """
    A router that takes products from an input buffer and routes them 
    to appropriate output buffers based on their next machine type.
    """
    _id_counter = 0

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        in_buffer: Buffer,
        type_to_buffer: dict[str, Buffer],
    ):
        """
        Initialize a Router.
        
        Args:
            env: SimPy environment
            name: Name of the router
            in_buffer: Input buffer to receive products from
            type_to_buffer: Mapping from machine type name to output buffer
        """
        self.env = env
        self.id = Router._id_counter
        Router._id_counter += 1
        self.name = name
        self.in_buffer = in_buffer
        self.type_to_buffer = type_to_buffer
        
        self.routed = 0
        self.proc = env.process(self._run())
    
    def _log(self, *args) -> None:
        """Log a message with timestamp and router name."""
        print(f"[{self.env.now:8.1f}] [RTR:{self.name}]", *args)
    
    def _run(self):
        """Generator process that routes products."""
        while True:
            # Get product from input buffer
            product: Product = yield self.in_buffer.get()
            
            # Determine next machine type
            next_type = product.next_machine_type()
            
            if next_type is None:
                # Product is complete, route to sink if available
                sink_key = "__SINK__"
                if sink_key in self.type_to_buffer:
                    self._log(f"Routing completed product {product.product_id} to sink")
                    yield self.type_to_buffer[sink_key].put(product)
                else:
                    self._log(f"WARNING: Product {product.product_id} complete but no sink available")
            else:
                # Route to appropriate buffer for next machine type
                if next_type in self.type_to_buffer:
                    self._log(f"Routing product {product.product_id} to {next_type} buffer")
                    yield self.type_to_buffer[next_type].put(product)
                    self.routed += 1
                else:
                    self._log(f"ERROR: No buffer for machine type '{next_type}'")

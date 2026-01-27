import simpy
from typing import Optional

from kata.entities.sources.base import Source as SourceBase
from kata.entities.buffers.base import Buffer
from kata.entities.products.product import Product


class Source(SourceBase):
    """A source that generates products and puts them into an output buffer."""
    
    def __init__(
        self,
        env: simpy.Environment,
        source_id: int,
        name: str,
        out_buffer: Buffer,
        interarrival_time: float = 10.0,
        route: Optional[list[str]] = None,
        max_products: Optional[int] = None,
    ):
        """
        Initialize a Source.
        
        Args:
            env: SimPy environment
            source_id: Unique identifier
            name: Name of the source
            out_buffer: Output buffer to send products to
            interarrival_time: Time between product arrivals
            route: Default route for products (list of machine type names)
            max_products: Maximum number of products to generate (None = infinite)
        """
        self.env = env
        self.id = source_id
        self.name = name
        self.out_buffer = out_buffer
        self.interarrival_time = interarrival_time
        self.route = route or []
        self.max_products = max_products
        
        self.products_created = 0
        self.proc = env.process(self._run())
    
    def _log(self, *args) -> None:
        """Log a message with timestamp and source name."""
        print(f"[{self.env.now:8.1f}] [SRC:{self.name}]", *args)
    
    def _run(self):
        """Generator process that creates products at regular intervals."""
        while True:
            # Check if we've reached the maximum
            if self.max_products is not None and self.products_created >= self.max_products:
                self._log(f"Reached max products ({self.max_products}), stopping")
                break
            
            # Wait for next product arrival
            yield self.env.timeout(self.interarrival_time)
            
            # Create product
            product = Product(
                product_id=self.products_created,
                route=self.route.copy(),
            )
            self.products_created += 1
            
            # Put into output buffer
            self._log(f"Creating product {product.product_id} with route {product.route}")
            yield self.out_buffer.put(product)
            self._log(f"Product {product.product_id} sent to {self.out_buffer.name}")

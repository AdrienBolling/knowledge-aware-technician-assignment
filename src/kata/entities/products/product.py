from typing import Optional


class Product:
    """A product flowing through the production line with a routing."""
    
    def __init__(self, product_id: int, route: list[str]):
        """
        Initialize a Product.
        
        Args:
            product_id: Unique identifier for this product
            route: List of machine type names that this product must visit
        """
        self.product_id = product_id
        self.route = route  # list of machine_type names
        self.step = 0  # Current step in the route
    
    def next_machine_type(self) -> Optional[str]:
        """Get the next machine type in the route."""
        return self.route[self.step] if self.step < len(self.route) else None
    
    def advance(self) -> None:
        """Advance to the next step in the route."""
        self.step += 1
    
    def is_complete(self) -> bool:
        """Check if the product has completed its route."""
        return self.step >= len(self.route)

import simpy
from typing import Any

from kata.entities.buffers.base import Buffer as BufferBase


class Buffer(BufferBase):
    """Concrete implementation of a SimPy buffer using simpy.Store."""
    
    def __init__(
        self,
        env: simpy.Environment,
        buffer_id: int,
        name: str,
        capacity: int = float('inf'),
    ):
        """
        Initialize a Buffer.
        
        Args:
            env: SimPy environment
            buffer_id: Unique identifier for this buffer
            name: Name of the buffer
            capacity: Maximum capacity of the buffer (default: infinite)
        """
        self.env = env
        self.id = buffer_id
        self.name = name
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=int(capacity) if capacity != float('inf') else None)
    
    def put(self, item: Any) -> simpy.Event:
        """
        Put an item into the buffer.
        
        Args:
            item: The item to put into the buffer
            
        Returns:
            SimPy event that succeeds when the item is placed
        """
        return self.store.put(item)
    
    def get(self) -> simpy.Event:
        """
        Get an item from the buffer.
        
        Returns:
            SimPy event that succeeds with the item when available
        """
        return self.store.get()
    
    def __len__(self) -> int:
        """
        Get the current number of items in the buffer.
        
        Returns:
            Number of items currently in the buffer
        """
        return len(self.store.items)

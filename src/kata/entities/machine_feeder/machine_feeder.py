import simpy

from kata.entities.machine_feeder.base import MachineFeeder as MachineFeederBase
from kata.entities.buffers.base import Buffer


class MachineFeeder(MachineFeederBase):
    """
    A feeder that distributes products from a type-specific buffer
    to multiple machines of that type (load balancing).
    """
    
    def __init__(
        self,
        env: simpy.Environment,
        feeder_id: int,
        name: str,
        machine_type: str,
        in_buffer: Buffer,
        machine_input_buffers: list[Buffer],
    ):
        """
        Initialize a MachineFeeder.
        
        Args:
            env: SimPy environment
            feeder_id: Unique identifier
            name: Name of the feeder
            machine_type: Type of machines this feeder serves
            in_buffer: Input buffer (type-specific queue)
            machine_input_buffers: List of input buffers for individual machines
        """
        self.env = env
        self.id = feeder_id
        self.name = name
        self.machine_type = machine_type
        self.in_buffer = in_buffer
        self.machine_input_buffers = machine_input_buffers
        
        self.fed = 0
        self.current_machine_idx = 0  # Round-robin index
        self.proc = env.process(self._run())
    
    def _log(self, *args) -> None:
        """Log a message with timestamp and feeder name."""
        print(f"[{self.env.now:8.1f}] [FEEDER:{self.name}]", *args)
    
    def _run(self):
        """Generator process that feeds products to machines."""
        while True:
            # Get product from type-specific buffer
            product = yield self.in_buffer.get()
            
            # Find next available machine buffer (round-robin with availability check)
            attempts = 0
            max_attempts = len(self.machine_input_buffers) * 2
            
            while attempts < max_attempts:
                target_buffer = self.machine_input_buffers[self.current_machine_idx]
                
                # Try to put into machine buffer (non-blocking check would be better)
                # For simplicity, just send to current machine in round-robin
                self._log(f"Feeding product {product.product_id} to {target_buffer.name}")
                yield target_buffer.put(product)
                self.fed += 1
                
                # Move to next machine for next product
                self.current_machine_idx = (self.current_machine_idx + 1) % len(self.machine_input_buffers)
                break

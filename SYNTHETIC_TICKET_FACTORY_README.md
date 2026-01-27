# SyntheticTicketFactory Documentation

## Overview

The `SyntheticTicketFactory` is a factory class designed to generate maintenance tickets from machine failure events in SimPy-based manufacturing simulations. It creates `SyntheticTicket` objects that capture comprehensive information about machine failures, including component details, operational context, and dynamic priority calculations.

## Features

### Core Functionality

1. **Ticket Generation from Repair Requests**
   - Automatically extracts machine and component information from `RepairRequest` objects
   - Links tickets to failed machines with full context

2. **Dynamic Field Population**
   - `machine`: Reference to the failed machine instance
   - `machine_type`: Type/category from `Machine.mtype`
   - `failure_type`: Inferred from component failure or set as general failure
   - `priority`: Dynamically calculated based on multiple factors
   - `nb_in_buffer`: Real-time buffer level from SimPy Store
   - `component_id` / `component_type`: Component details (for ComplexMachine)
   - `repair_time_estimate`: Expected repair duration

3. **Intelligent Priority Calculation**
   - Component-specific base priorities via configurable rules
   - Buffer level adjustment (more items = higher priority)
   - Machine productivity consideration
   - Optional random variance for simulation realism

4. **Batch Processing**
   - Generate multiple tickets from a list of repair requests
   - Efficient for bulk operations

5. **Integration with ComplexMachine**
   - Seamlessly extracts component failure information
   - Respects component-specific repair times
   - Provides detailed failure type classification

## Installation

The SyntheticTicketFactory is part of the `kata` package. Ensure all dependencies are installed:

```bash
pip install simpy numpy pydantic pydantic-settings gymnasium numba
```

## Quick Start

### Basic Usage

```python
from kata.EntityFactories.synthetic_ticket_factory import SyntheticTicketFactory
from kata.entities.requests.RepairRequest import RepairRequest

# Create factory
factory = SyntheticTicketFactory()

# Generate ticket from a repair request
ticket = factory.create_ticket_from_repair_request(repair_request)

# Access ticket information
print(f"Machine: {ticket.get_machine_id()}")
print(f"Priority: {ticket.get_priority()}")
print(f"Buffer Level: {ticket.get_buffer_level()}")
print(f"Failure Type: {ticket.get_failure_type()}")
```

### Custom Priority Rules

```python
# Define component-specific priorities
priority_rules = {
    "motor": 10,      # Critical
    "sensor": 3,      # Less critical
    "pump": 7,        # Moderate
    "bearing": 8,     # Important
}

factory = SyntheticTicketFactory(priority_rules=priority_rules)
ticket = factory.create_ticket_from_repair_request(repair_request)
```

### Adding Randomness

```python
# Enable random variance in priority calculations
factory = SyntheticTicketFactory(
    add_randomness=True,
    random_priority_variance=2,  # +/- 2 priority points
)
```

### Batch Generation

```python
# Generate multiple tickets at once
repair_requests = [request1, request2, request3]
tickets = factory.create_batch_tickets(repair_requests)

for ticket in tickets:
    print(f"Ticket {ticket.ticket_id}: Priority {ticket.get_priority()}")
```

## API Reference

### SyntheticTicket

#### Constructor

```python
SyntheticTicket(
    machine: Machine,
    machine_type: str,
    failure_type: str,
    priority: int,
    nb_in_buffer: int,
    created_at: int,
    ticket_id: Optional[int] = None,
    component_id: Optional[str] = None,
    component_type: Optional[str] = None,
    repair_time_estimate: Optional[float] = None,
)
```

#### Methods

- `get_machine_id() -> int`: Returns the failed machine's ID
- `get_machine_type() -> str`: Returns the machine type
- `get_failure_type() -> str`: Returns the failure classification
- `get_priority() -> int`: Returns the priority level
- `get_buffer_level() -> int`: Returns the buffer item count
- `get_component_info() -> Optional[dict]`: Returns component details (if applicable)

### SyntheticTicketFactory

#### Constructor

```python
SyntheticTicketFactory(
    priority_rules: Optional[dict] = None,
    add_randomness: bool = False,
    random_priority_variance: int = 0,
    ticket_id_counter: int = 1,
)
```

**Parameters:**
- `priority_rules`: Dictionary mapping failure/component types to base priorities
- `add_randomness`: Enable random variance in priority calculations
- `random_priority_variance`: Range of random adjustment (+/- value)
- `ticket_id_counter`: Starting value for ticket ID generation

#### Methods

##### create_ticket_from_repair_request

```python
create_ticket_from_repair_request(repair_request: RepairRequest) -> SyntheticTicket
```

Generate a ticket from a repair request. This is the primary method for integration with the simulation's repair workflow.

**Returns:** `SyntheticTicket` instance

##### create_ticket_from_machine

```python
create_ticket_from_machine(machine: Machine, created_at: int) -> SyntheticTicket
```

Generate a ticket directly from a machine instance. Useful for custom workflows.

**Parameters:**
- `machine`: The failed machine
- `created_at`: Simulation time of ticket creation

**Returns:** `SyntheticTicket` instance

##### create_batch_tickets

```python
create_batch_tickets(repair_requests: List[RepairRequest]) -> List[SyntheticTicket]
```

Generate multiple tickets from a list of repair requests.

**Returns:** List of `SyntheticTicket` instances

## Priority Calculation Logic

The priority calculation considers multiple factors:

1. **Base Priority**: From `priority_rules` based on component/failure type (default: 5)
2. **Buffer Adjustment**:
   - Buffer > 10 items: +3
   - Buffer > 5 items: +2
   - Buffer > 0 items: +1
3. **Productivity Adjustment**:
   - Total processed > 100: +2
   - Total processed > 50: +1
4. **Random Variance** (if enabled): +/- configured variance
5. **Minimum Priority**: Always at least 1

### Example Calculation

```python
priority_rules = {"motor": 10}
# Motor component failed
# Buffer has 12 items (>10, so +3)
# Machine processed 120 items (>100, so +2)
# Result: 10 + 3 + 2 = 15
```

## Integration Examples

### With GymTechDispatcher

```python
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.EntityFactories.synthetic_ticket_factory import SyntheticTicketFactory

# In your simulation setup
factory = SyntheticTicketFactory(priority_rules={
    "motor": 10,
    "pump": 7,
    "sensor": 3,
})

# When a machine breaks down, the dispatcher creates a RepairRequest
# Generate a ticket for logging/analysis
def on_machine_breakdown(repair_request):
    ticket = factory.create_ticket_from_repair_request(repair_request)
    
    # Log or store the ticket
    print(f"Maintenance Ticket #{ticket.ticket_id} created")
    print(f"  Machine: {ticket.get_machine_id()}")
    print(f"  Priority: {ticket.get_priority()}")
    
    # Could store in database, send to scheduler, etc.
    save_to_database(ticket)
```

### Simulation Loop Integration

```python
def run_simulation(env, machines, factory):
    """Example simulation with ticket generation."""
    repair_requests = []
    
    # Collect repair requests during simulation
    while env.now < simulation_duration:
        # ... simulation runs ...
        
        # When machines fail, requests are added to queue
        if not tech_dispatcher.repair_queue.items:
            yield env.timeout(1)
            continue
        
        # Process pending requests
        request = yield tech_dispatcher.repair_queue.get()
        repair_requests.append(request)
        
        # Generate ticket
        ticket = factory.create_ticket_from_repair_request(request)
        
        # Dispatch technician based on priority
        dispatch_technician_by_priority(ticket)
    
    # Generate batch report at end
    all_tickets = factory.create_batch_tickets(repair_requests)
    generate_maintenance_report(all_tickets)
```

## Testing

Run the test suite to verify functionality:

```bash
PYTHONPATH=src:$PYTHONPATH python tests/test_synthetic_ticket_factory.py
```

Run the demonstration script:

```bash
PYTHONPATH=src:$PYTHONPATH python demo_synthetic_ticket_factory.py
```

## Advanced Usage

### Custom Priority Logic

For more complex priority rules, you can subclass `SyntheticTicketFactory`:

```python
class CustomTicketFactory(SyntheticTicketFactory):
    def _calculate_priority(self, machine, failure_type, component_type=None):
        # Custom logic
        priority = super()._calculate_priority(machine, failure_type, component_type)
        
        # Add time-of-day consideration
        if 8 <= machine.env.now % 24 <= 17:  # Business hours
            priority += 3
        
        # Add machine criticality from config
        if hasattr(machine, 'criticality_level'):
            priority += machine.criticality_level
        
        return priority
```

### Ticket Analytics

```python
def analyze_tickets(tickets):
    """Analyze a collection of tickets."""
    priorities = [t.get_priority() for t in tickets]
    buffer_levels = [t.get_buffer_level() for t in tickets]
    
    print(f"Total Tickets: {len(tickets)}")
    print(f"Average Priority: {sum(priorities) / len(priorities):.1f}")
    print(f"Average Buffer Level: {sum(buffer_levels) / len(buffer_levels):.1f}")
    
    # Group by machine type
    by_type = {}
    for ticket in tickets:
        mtype = ticket.get_machine_type()
        by_type[mtype] = by_type.get(mtype, 0) + 1
    
    print(f"Tickets by Machine Type: {by_type}")
```

## Best Practices

1. **Define Priority Rules Early**: Set up your priority rules based on domain knowledge
2. **Use Batch Generation for Reports**: When analyzing simulation results, use batch generation for efficiency
3. **Monitor Buffer Levels**: The buffer-based priority adjustment helps identify bottlenecks
4. **Component-Specific Rules**: For ComplexMachine, define rules for each component type
5. **Test Priority Calculations**: Verify that priority calculations align with business needs

## Troubleshooting

### Issue: Ticket has priority 0 or negative

**Solution**: Priority is always at least 1. Check your `priority_rules` and `random_priority_variance` settings.

### Issue: Buffer level always shows 0

**Solution**: Ensure your machine's `input_buffer` is properly initialized and has a `items` attribute or implements `__len__`.

### Issue: Component information is None for ComplexMachine

**Solution**: Verify that the machine's `failed_component` is set before creating the RepairRequest.

## License

See repository LICENSE file.

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting pull requests.

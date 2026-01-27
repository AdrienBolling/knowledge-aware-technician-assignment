import simpy


class Technician:
    """Basic Technician implementation for SimPy simulation."""
    
    def __init__(
        self,
        tech_id: int,
        name: str,
        travel_time: int = 10,
        efficiency: float = 1.0,
    ):
        """
        Initialize a Technician.
        
        Args:
            tech_id: Unique identifier for this technician
            name: Name of the technician
            travel_time: Time to travel to any machine (simplified)
            efficiency: Multiplier for repair time (< 1.0 = faster, > 1.0 = slower)
        """
        self.id = tech_id
        self.name = name
        self.travel_time_value = travel_time
        self.efficiency = efficiency
        self.busy = False


class TechDispatcher:
    def __init__(
        self,
        env: simpy.Environment,
        technicians: list[Technician],
    ) -> None:
        self.env = env
        self.techs = technicians

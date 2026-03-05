import simpy


class Technician:
    """Basic Technician implementation for SimPy simulation."""
    _id_counter = 0

    def __init__(
        self,
        name: str,
        travel_time: int = 10,
        efficiency: float = 1.0,
    ):
        """
        Initialize a Technician.
        
        Args:
            name: Name of the technician
            travel_time: Time to travel to any machine (simplified)
            efficiency: Multiplier for repair time (< 1.0 = faster, > 1.0 = slower)
        """
        self.id = Technician._id_counter
        Technician._id_counter += 1
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

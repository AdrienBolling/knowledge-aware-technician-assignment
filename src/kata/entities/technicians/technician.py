import simpy


class Technician:
    pass


class TechDispatcher:
    def __init__(
        self,
        env: simpy.Environment,
        technicians: list[Technician],
    ) -> None:
        self.env = env
        self.techs = technicians

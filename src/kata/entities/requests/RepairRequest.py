from kata.entities import Request


class RepairRequest(Request):
    chosen_technician_id: int | None = None

    def __init__(self, machine, created_at: int) -> None:
        pass

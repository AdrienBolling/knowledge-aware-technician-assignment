import simpy
from typing import Any


class Buffer:
    env: simpy.Environment
    id: int
    name: str

    def put(self, item) -> None:
        raise NotImplementedError

    def get(self) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

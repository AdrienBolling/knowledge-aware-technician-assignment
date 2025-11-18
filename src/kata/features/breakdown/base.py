class BreakdownProcess:
    def step_and_get_proba(self) -> float:
        raise NotImplementedError()

    def step_and_get_idle_proba(self) -> float:
        raise NotImplementedError()

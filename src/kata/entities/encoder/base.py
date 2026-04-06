"""Base class for the RepairRequest encoder."""

from kata.entities.requests.RepairRequest import RepairRequest
import prince
import numpy as np

class Encoder:
    """The basic encoder is based on prince's MCA (basically PCA for categorical data)."""

    def __init__(self) -> None :

        self.mca = prince.MCA(
            n_components = 2,
            n_iter = 10,
            copy = True,
            check_input = True,
            engine = 'sklearn',
            random_state = 42,
            correction = 'benzecri',
        )

    def fit(self, dataset) -> None:

        self.mca.fit(dataset)

    def _requests_to_dataset(self, requests: list[RepairRequest]) -> np.ndarray:
        

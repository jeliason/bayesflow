from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Normalize(ElementwiseTransform):
    """
    Transform that when applied standardizes data using typical z-score standardization i.e. for some unstandardized
    data x the standardized version z  would be

    z = (x - min(x))/(max(x) - min(x))

    """

    def __init__(
        self,
        min: int | float | np.ndarray = None,
        max: int | float | np.ndarray = None,
        axis: int = None,
        eps: float = 1e-8
    ):
        super().__init__()

        self.min = min
        self.max = max
        self.axis = axis
        self.eps = eps  # Small value to prevent division by zero


    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Normalize":
        return cls(
            min=deserialize(config["min"], custom_objects),
            max=deserialize(config["max"], custom_objects),
            axis=deserialize(config["axis"], custom_objects)
        )

    def get_config(self) -> dict:
        return {
            "min": serialize(self.min),
            "max": serialize(self.max),
            "axis": serialize(self.axis)
        }

    def forward(self, data: np.ndarray, stage: str = "training", **kwargs) -> np.ndarray:
        if self.axis is None:
            self.axis = tuple(range(data.ndim - 1))

        if self.min is None:
            self.min = np.nanmin(data, axis=self.axis, keepdims=True)

        if self.max is None:
            self.max = np.nanmax(data, axis=self.axis, keepdims=True)


        min = np.broadcast_to(self.min, data.shape)
        max = np.broadcast_to(self.max, data.shape)

        return (data - min) / (max - min + self.eps)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.min is None or self.max is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        min = np.broadcast_to(self.min, data.shape)
        max = np.broadcast_to(self.max, data.shape)

        return data * (max - min + self.eps) + min

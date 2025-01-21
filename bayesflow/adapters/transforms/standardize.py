from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Standardize(ElementwiseTransform):
    """
    Transform that when applied standardizes data using typical z-score standardization i.e. for some unstandardized
    data x the standardized version z  would be

    z = (x - mean(x))/std(x)

    Parameters:
    mean: integer or float used to specify a mean if known but will be estimated from data when not provided
    std: integer or float used to specify a standard devation if known but will be estimated from data when not provided
    axis: integer representing a specific axis along which standardization should take place. By default
        standardization happens individually for each dimension
    momentum: float in (0,1) specifying the momentum during training

    """

    def __init__(
        self,
        mean: int | float | np.ndarray = None,
        std: int | float | np.ndarray = None,
        axis: int = None,
        momentum: float | None = 0.99,
    ):
        super().__init__()

        self.mean = mean
        self.std = std
        self.axis = axis
        self.momentum = momentum

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Standardize":
        return cls(
            mean=deserialize(config["mean"], custom_objects),
            std=deserialize(config["std"], custom_objects),
            axis=deserialize(config["axis"], custom_objects),
            momentum=deserialize(config["momentum"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "mean": serialize(self.mean),
            "std": serialize(self.std),
            "axis": serialize(self.axis),
            "momentum": serialize(self.momentum),
        }

    def forward(self, data: np.ndarray, stage: str = "training", **kwargs) -> np.ndarray:
        if self.axis is None:
            self.axis = tuple(range(data.ndim - 1))

        if self.mean is None:
            self.mean = np.nanmean(data, axis=self.axis, keepdims=True)
        else:
            if self.momentum is not None and stage == "training":
                self.mean = self.momentum * self.mean + (1.0 - self.momentum) * np.mean(
                    data, axis=self.axis, keepdims=True
                )

        if self.std is None:
            self.std = np.nanstd(data, axis=self.axis, keepdims=True, ddof=1)
        else:
            if self.momentum is not None and stage == "training":
                self.std = self.momentum * self.std + (1.0 - self.momentum) * np.std(
                    data, axis=self.axis, keepdims=True, ddof=1
                )

        eps = 1e-8  # Small value to prevent division by zero

        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape) + eps

        return (data - mean) / std

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return data * std + mean

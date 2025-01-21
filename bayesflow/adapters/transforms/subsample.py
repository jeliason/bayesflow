from collections.abc import Sequence, Callable

from keras.saving import (
		deserialize_keras_object as deserialize,
		register_keras_serializable as serializable,
		serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.adapters")
class Subsample(Transform):
						
		def __init__(self, keys: Sequence[str], *, sampler: Callable[[int],Sequence[int]] = None, axes: Sequence[int] = None):
				super().__init__()

				self.keys = keys
				self.sampler = sampler
				self.axes = axes

		@classmethod
		def from_config(cls, config: dict, custom_objects=None) -> "Subsample":
				return cls(keys=deserialize(config["keys"], custom_objects),
										sampler=deserialize(config["sampler"], custom_objects),
										axes=deserialize(config["axes"], custom_objects))

		def get_config(self) -> dict:
				return {"keys": serialize(self.keys),
								"sampler": serialize(self.sampler),
								"axes": serialize(self.axes)}

		def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
				# generate random index
				total_time_points = data[self.keys[0]].shape[self.axes[0]]
				random_index = self.sampler(total_time_points)
				# subsample data
				for ax,key in zip(self.axes,self.keys):
						arr = data[key]
						index = [slice(None)] * arr.ndim  # Create a list of slices
						index[ax] = random_index  # Replace the specific axis with the index
						sliced_arr = arr[tuple(index)]  # Efficient slicing (returns a view when possible)
						data[key] = sliced_arr
				return data

		def inverse(self, data: dict[str, any], **kwargs) -> dict[str, any]:
				# non-invertible transform (for now)
				return data

import math

from keras import ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.utils import searchsorted
from bayesflow.types import Tensor

from .transform import Transform


@serializable(package="networks.coupling_flow")
class SplineTransform(Transform):
    def __init__(self, bins=16, default_domain=(-5.0, 5.0, -5.0, 5.0), **kwargs):
        super().__init__(**kwargs)

        self.bins = bins
        self.default_domain = default_domain
        self.spline_params_counts = {
            "left_edge": 1,
            "bottom_edge": 1,
            "widths": self.bins,
            "heights": self.bins,
            "derivatives": self.bins - 1,
        }
        self.split_idx = ops.cumsum(list(self.spline_params_counts.values()))[:-1]
        self._params_per_dim = sum(self.spline_params_counts.values())

        # Pre-compute defaults and softplus shifts
        default_width = (self.default_domain[1] - self.default_domain[0]) / self.bins
        default_height = (self.default_domain[3] - self.default_domain[2]) / self.bins
        self.xshift = math.log(math.exp(default_width) - 1)
        self.yshift = math.log(math.exp(default_height) - 1)
        self.softplus_shift = math.log(math.e - 1.0)

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        # Ensure spline works for N-D, e.g., 2D (batch_size, dim) and 3D (batch_size, num_reps, dim)
        shape = ops.shape(parameters)
        new_shape = shape[:-1] + (-1, self._params_per_dim)

        # Arrange spline parameters into a dictionary
        parameters = ops.reshape(parameters, new_shape)
        parameters = ops.split(parameters, self.split_idx, axis=-1)
        parameters = dict(
            left_edge=parameters[0],
            bottom_edge=parameters[1],
            widths=parameters[2],
            heights=parameters[3],
            derivatives=parameters[4],
        )
        return parameters

    @property
    def params_per_dim(self):
        return self._params_per_dim

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        # Set lower corners of domain relative to default domain
        parameters["left_edge"] = parameters["left_edge"] + self.default_domain[0]
        parameters["bottom_edge"] = parameters["bottom_edge"] + self.default_domain[2]

        # Constrain widths and heights to be positive
        parameters["widths"] = ops.softplus(parameters["widths"] + self.xshift)
        parameters["heights"] = ops.softplus(parameters["heights"] + self.yshift)

        # Compute spline derivatives
        parameters["derivatives"] = ops.softplus(parameters["derivatives"] + self.softplus_shift)

        # Add in edge derivatives
        total_width = ops.sum(parameters["widths"], axis=-1, keepdims=True)
        total_height = ops.sum(parameters["heights"], axis=-1, keepdims=True)
        scale = total_height / total_width
        parameters["derivatives"] = ops.concatenate([scale, parameters["derivatives"], scale], axis=-1)
        return parameters

    def _forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        return self._calculate_spline(x, parameters, inverse=False)

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        return self._calculate_spline(z, parameters, inverse=True)

    @staticmethod
    def _calculate_spline(x: Tensor, p: dict[str, Tensor], inverse: bool = False) -> (Tensor, Tensor):
        """Helper function to calculate RQ spline."""

        result = ops.zeros_like(x)
        log_jac = ops.zeros_like(x)

        total_width = ops.sum(p["widths"], axis=-1, keepdims=True)
        total_height = ops.sum(p["heights"], axis=-1, keepdims=True)

        knots_x = ops.concatenate([p["left_edge"], p["left_edge"] + ops.cumsum(p["widths"], axis=-1)], axis=-1)
        knots_y = ops.concatenate([p["bottom_edge"], p["bottom_edge"] + ops.cumsum(p["heights"], axis=-1)], axis=-1)

        if not inverse:
            target_in_domain = ops.logical_and(knots_x[..., 0] < x, x <= knots_x[..., -1])
            higher_indices = searchsorted(knots_x, x[..., None])
        else:
            target_in_domain = ops.logical_and(knots_y[..., 0] < x, x <= knots_y[..., -1])
            higher_indices = searchsorted(knots_y, x[..., None])

        target_in = x[target_in_domain]
        target_in_idx = ops.stack(ops.where(target_in_domain), axis=-1)
        target_out = x[~target_in_domain]
        target_out_idx = ops.stack(ops.where(~target_in_domain), axis=-1)

        # In-domain computation
        if ops.size(target_in_idx) > 0:
            # Index crunching
            higher_indices = ops.take_along_axis(higher_indices, target_in_idx)
            lower_indices = higher_indices - 1
            lower_idx_tuples = ops.concatenate([target_in_idx, lower_indices], axis=-1)
            higher_idx_tuples = ops.concatenate([target_in_idx, higher_indices], axis=-1)

            # Spline computation
            dk = ops.take_along_axis(p["derivatives"], lower_idx_tuples)
            dkp = ops.take_along_axis(p["derivatives"], higher_idx_tuples)
            xk = ops.take_along_axis(knots_x, lower_idx_tuples)
            xkp = ops.take_along_axis(knots_x, higher_idx_tuples)
            yk = ops.take_along_axis(knots_y, lower_idx_tuples)
            ykp = ops.take_along_axis(knots_y, higher_idx_tuples)
            x = target_in
            dx = xkp - xk
            dy = ykp - yk
            sk = dy / dx
            xi = (x - xk) / dx

            # Forward pass
            if not inverse:
                numerator = dy * (sk * xi**2 + dk * xi * (1 - xi))
                denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
                result_in = yk + numerator / denominator

                # Log Jacobian for in-domain
                numerator = sk**2 * (dkp * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
                denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
                log_jac_in = ops.log(numerator + 1e-10) - ops.log(denominator + 1e-10)
                log_jac = ops.slice_update(log_jac, target_in_idx, log_jac_in)

            # Inverse pass
            else:
                y = x
                a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
                b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
                c = -sk * (y - yk)
                discriminant = ops.maximum(b**2 - 4 * a * c, 0.0)
                xi = 2 * c / (-b - ops.sqrt(discriminant))
                result_in = xi * dx + xk

            result = ops.slice_update(result, target_in_idx, result_in)

        # Out-of-domain
        if ops.size(target_out_idx) > 1:
            scale = total_height / total_width
            shift = p["bottom_edge"] - scale * p["left_edge"]
            scale_out = ops.take_along_axis(scale, target_out_idx)
            shift_out = ops.take_along_axis(shift, target_out_idx)

            if not inverse:
                result_out = scale_out * target_out[..., None] + shift_out
                # Log Jacobian for out-of-domain points
                log_jac_out = ops.log(scale_out + 1e-10)
                log_jac_out = ops.squeeze(log_jac_out, axis=-1)
                log_jac = ops.slice_update(log_jac, target_out_idx, log_jac_out)
            else:
                result_out = (target_out[..., None] - shift_out) / scale_out

            result_out = ops.squeeze(result_out, axis=-1)
            result = ops.slice_update(result, target_out_idx, result_out)

        log_det = ops.sum(log_jac, axis=-1)
        return result, log_det

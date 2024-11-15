
from collections.abc import Callable, Mapping, Sequence
from functools import partial
import keras
import numpy as np

from bayesflow.types import Tensor
from optree import PyTree

ArrayLike = int | float | Tensor
Integrand = Callable[[PyTree[Tensor], ...], PyTree[Tensor]]


def compute_deltas(..., method: str = "euler") -> PyTree[Tensor]:
    ...


def update_integrands(integrands: PyTree[Tensor], deltas: PyTree[Tensor]) -> PyTree[Tensor]:



def integrate(
        fn: Integrand,
        integrands: PyTree[Tensor],
        start_time: PyTree[ArrayLike],
        stop_time: PyTree[ArrayLike],
        method: str = "euler",
        steps: int = "adaptive",
        step_size: ArrayLike = "adaptive",
        **kwargs
) -> list[ArrayLike]:

    results = list(args)

    # TODO: shouldn't start_time, stop_time etc. technically be Sequence[ArrayLike]?
    #  also, implement the select_initial_step_size from here: https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/common.py#L68
    #  also, some args may not be inputs to fn, and some may not be outputs. Use something like a callback?

    match method:
        case "euler":
            step_fn = euler_step
        case "rk45":
            step_fn = rk45_step
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    match steps, step_size:
        case "adaptive", "adaptive":
            use_adaptive_step_size = True
            step_size = select_initial_step_size(...)
        case "adaptive", step_size:
            use_adaptive_step_size = True
        case int(), "adaptive":
            use_adaptive_step_size = False
            step_size = (stop_time - start_time) / steps
        case int(), _:
            raise ValueError("Cannot specify both `steps` and `step_size`.")
        case _:
            raise RuntimeError("Type or value of `steps` or `step_size` not understood.")

    time = start_time
    while np.any(time < stop_time):
        results, time, step_size = step_fn(
            fn,
            *results,
            time=time,
            step_size=step_size,
            use_adaptive_step_size=use_adaptive_step_size,
            **kwargs,
        )

    return results


def select_initial_step_size(fn: Integrand, *args: ArrayLike, start_time: ArrayLike, stop_time: ArrayLike, max_step_size: ArrayLike, rtol: ArrayLike, atol: ArrayLike) -> ArrayLike:
    """ Analog of scipy.integrate._ivp.common.select_initial_step """
    k1 = fn(start_time, *args)
    scales = [atol + keras.ops.abs(a) * rtol for a in args]
    ...

    return (stop_time - start_time) / 1000




def euler_step(
        fn: Integrand,
        *args: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        use_adaptive_step_size: bool = True,
        max_step_size: ArrayLike = float("inf"),
        tolerance: ArrayLike = 1e-6,
) -> (Sequence[ArrayLike], ArrayLike):
    results = list(args)

    k1 = fn(time, *results)

    if use_adaptive_step_size:
        k2 = fn(time + step_size, *[r + step_size * k for r, k in zip(results, k1)])
        error = keras.ops.stack([keras.ops.norm(k2[i] - k1[i]) for i in range(len(results))])
        step_size = keras.ops.minimum(step_size * tolerance / error, max_step_size)

    for i in range(len(results)):
        results[i] += step_size * k1[i]

    time += step_size

    return results, time, step_size


def rk45_step(
        fn: Integrand,
        *integrands: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike | str,
        max_step_size: ArrayLike,
        tolerance: ArrayLike,
) -> (Sequence[ArrayLike], ArrayLike):
    results = list(integrands)

    k1 = fn(time, *results)
    k2 = fn(time + 0.5 * step_size, *[r + 0.5 * step_size * k for r, k in zip(results, k1)])
    k3 = fn(time + 0.5 * step_size, *[r + 0.5 * step_size * k for r, k in zip(results, k2)])
    k4 = fn(time + step_size, *[r + step_size * k for r, k in zip(results, k3)])

    if step_size == "adaptive":
        k5 = fn(time + 0.5 * step_size, *[r + 0.5 * step_size * k for r, k in zip(results, k4)])
        error = keras.ops.stack([keras.ops.norm(k5[i] - k4[i]) for i in range(len(results))])
        step_size = keras.ops.minimum(step_size * tolerance / error, max_step_size)

    for i in range(len(results)):
        results[i] += (step_size / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])

    time += step_size

    return results, time, step_size

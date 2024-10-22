from bayesflow.types import Shape


def validate_batch_shape(batch_shape: Shape) -> tuple:
    if isinstance(batch_shape, int):
        batch_shape = (batch_shape,)

    return batch_shape

from typing import Any

from auto_cast.types import Batch


class Preprocessor:
    """Base Preprocessor.

    This is not trainable but can combine the elements of the batch into a form that
    can be passed to the call/forward of the models.

    """

    def __call__(self, x: Batch) -> Any:
        """Forward Pass through the Preprocessor."""
        msg = "To implement."
        raise NotImplementedError(msg)

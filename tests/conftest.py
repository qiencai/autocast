"""Shared test fixtures and utilities.

This conftest.py is automatically discovered by pytest and makes utilities
available to all test modules in this directory and subdirectories.

Current utilities are primarily for DCAE encoder/decoder testing but can be
extended for other modules.
"""

from torch import Tensor, nn


def assert_output_valid(output: Tensor, expected_shape: tuple, name: str = "Output"):
    """Assert output has expected shape and contains no NaN values."""
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )
    assert not output.isnan().any(), f"{name} contains NaN values"


def assert_module_initialized(module: nn.Module, module_name: str):
    """Assert module is properly initialized with valid parameters."""
    assert isinstance(module, nn.Module), f"{module_name} is not a Module"
    params = list(module.parameters())
    assert len(params) > 0, f"{module_name} has no parameters"
    for param in params:
        assert not param.isnan().any(), "Parameters contain NaN values"


def assert_gradients_flow(module: nn.Module, input_tensor: Tensor, module_name: str):
    """Assert gradients flow through module to input and all parameters."""
    output = module.forward(input_tensor)
    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert input_tensor.grad is not None, f"{module_name}: No gradients on input"
    assert not input_tensor.grad.isnan().any(), (
        f"{module_name}: Input gradients contain NaN"
    )

    # Check parameter gradients
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"{module_name}: No gradients on parameter {name}"
            )
            assert not param.grad.isnan().any(), (
                f"{module_name}: Parameter {name} gradients contain NaN"
            )

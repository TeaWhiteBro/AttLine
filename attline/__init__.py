"""Lightweight attention visualizer for diffusion pipelines."""
from .api import visualize_attention
from .attach import attach, detach
from .capture import CaptureState
from .patch import PipelineAdapter, register_adapter, supported_pipelines
from .selectors import Selection, TextMeta, resolve_selection

# Register built-in adapters (side-effect imports).
from . import _flux1_dev as _flux1_dev  # noqa: F401
from . import _flux2_klein as _flux2_klein  # noqa: F401
from . import _qwenimage_2512 as _qwenimage_2512  # noqa: F401

__version__ = "0.4.0"

__all__ = [
    "attach",
    "detach",
    "visualize_attention",
    "CaptureState",
    "PipelineAdapter",
    "Selection",
    "TextMeta",
    "register_adapter",
    "resolve_selection",
    "supported_pipelines",
    "__version__",
]

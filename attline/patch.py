"""Pipeline-agnostic patching for attention capture.

This module provides:

- ``PipelineAdapter`` — a descriptor that tells the system how to patch a
  specific diffusion pipeline class.
- ``register_adapter`` / ``supported_pipelines`` — adapter registry.
- ``patched_pipeline`` — context manager that swaps the pipeline ``__call__``
  *and* the native attention backend, runs the user's generation, then
  restores everything.
- ``_native_attention_capture_backend`` — the generic SDPA replacement that
  intercepts attention matrices for capture. This function is pipeline-
  agnostic; it works with any transformer that uses the ``NATIVE`` backend
  from ``diffusers.models.attention_dispatch``.

Pipeline-specific adapters live in separate modules (e.g. ``_flux2_klein``)
and register themselves via ``register_adapter`` at import time.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import importlib

import torch

from .capture import CaptureState, compute_attention_with_capture, get_capture_state, set_capture_state


# ---------------------------------------------------------------------------
# Adapter dataclass + registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineAdapter:
    """Describes how to patch a specific pipeline class for attention capture.

    Parameters
    ----------
    name
        Human-readable name, e.g. ``"Flux2KleinPipeline"``.
    pipeline_class_names
        One or more class names that this adapter handles. Used for
        auto-detection when the caller does not specify ``pipeline_type``.
    patched_call
        The replacement ``__call__`` method. It must have the same signature
        as the original pipeline ``__call__`` and additionally read the
        global ``CaptureState`` (via ``get_capture_state()``) to build the
        token layout and stash it on the state.
    """
    name: str
    pipeline_class_names: Tuple[str, ...]
    patched_call: Callable


_ADAPTERS: Dict[str, PipelineAdapter] = {}


def register_adapter(adapter: PipelineAdapter) -> None:
    """Register a pipeline adapter for attention capture.

    After registration the adapter can be looked up by its ``name`` or by
    any of its ``pipeline_class_names``.
    """
    _ADAPTERS[adapter.name] = adapter
    for cls_name in adapter.pipeline_class_names:
        _ADAPTERS[cls_name] = adapter


def supported_pipelines() -> List[str]:
    """Return sorted list of registered adapter names."""
    return sorted({a.name for a in _ADAPTERS.values()})


def _format_unsupported_error(pipeline_name: str) -> str:
    supported = supported_pipelines()
    return (
        f"Unsupported pipeline: {pipeline_name!r}.\n"
        f"Currently supported pipelines: {supported}.\n"
        "To add a new pipeline, write an adapter and call "
        "`register_adapter(PipelineAdapter(...))` — see "
        "`attline/_flux2_klein.py` for a reference implementation."
    )


def _find_adapter(
    pipe: Any,
    pipeline_type: Optional[str] = None,
) -> PipelineAdapter:
    if pipeline_type is not None:
        adapter = _ADAPTERS.get(pipeline_type)
        if adapter is None:
            raise ValueError(_format_unsupported_error(pipeline_type))
        return adapter

    # Auto-detect by walking the class MRO.
    for cls in type(pipe).__mro__:
        adapter = _ADAPTERS.get(cls.__name__)
        if adapter is not None:
            return adapter

    raise ValueError(_format_unsupported_error(type(pipe).__name__))


# ---------------------------------------------------------------------------
# Attention backend swap helper
# ---------------------------------------------------------------------------

def _swap_attention_backend() -> Callable[[], None]:
    """Replace the NATIVE attention backend with the capture-enabled version.

    Returns a *restore* callable that puts the original backend back.
    """
    dispatch = importlib.import_module("diffusers.models.attention_dispatch")
    Reg = dispatch._AttentionBackendRegistry
    Name = dispatch.AttentionBackendName

    original_backend = Reg._backends[Name.NATIVE]
    original_native = getattr(dispatch, "_native_attention", original_backend)

    dispatch._ORIG_NATIVE_ATTN_BACKEND = original_backend
    Reg._backends[Name.NATIVE] = _native_attention_capture_backend
    dispatch._native_attention = _native_attention_capture_backend

    def restore() -> None:
        Reg._backends[Name.NATIVE] = original_backend
        dispatch._native_attention = original_native

    return restore


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def patched_pipeline(
    pipe: Any,
    state: CaptureState,
    pipeline_type: Optional[str] = None,
) -> Generator[None, None, None]:
    """Context manager that patches *pipe* for attention capture.

    1. Looks up the ``PipelineAdapter`` for *pipe* (auto-detected or via
       *pipeline_type*).
    2. Replaces ``pipe.__class__.__call__`` with the adapter's patched call.
    3. Swaps the ``NATIVE`` attention backend for the capture-enabled version.
    4. Sets the global ``CaptureState``.
    5. On exit, restores everything.
    """
    adapter = _find_adapter(pipe, pipeline_type)
    pipeline_cls = pipe.__class__
    original_call = pipeline_cls.__call__

    pipeline_cls.__call__ = adapter.patched_call
    restore_backend = _swap_attention_backend()
    set_capture_state(state)

    try:
        yield
    finally:
        set_capture_state(None)
        restore_backend()
        pipeline_cls.__call__ = original_call


# ---------------------------------------------------------------------------
# Native attention capture backend (pipeline-agnostic)
# ---------------------------------------------------------------------------

def _native_attention_capture_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional[Any] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("Native attention backend does not support return_lse=True.")

    dispatch = importlib.import_module("diffusers.models.attention_dispatch")
    state = get_capture_state()
    original_backend = getattr(dispatch, "_ORIG_NATIVE_ATTN_BACKEND", None)

    if _parallel_config is not None or state is None:
        return original_backend(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse,
            _parallel_config=_parallel_config,
        )

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    try:
        out = compute_attention_with_capture(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            state=state,
        )
    except torch.cuda.OutOfMemoryError:
        if not state.fallback_to_sdpa:
            raise
        state.skipped_capture_calls += 1
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return out.permute(0, 2, 1, 3)

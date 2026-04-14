"""One-line ``attach`` / ``detach`` API for persistent attention capture.

Usage::

    from attline import attach

    attach(pipe, words=["cat", "hello"], save_dir="./attn_out")

    image = pipe(prompt=..., height=1024, ...).images[0]
    # Heatmaps already saved in ./attn_out/
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image

from .api import _build_pair_list
from .capture import CaptureState, set_capture_state
from .patch import PipelineAdapter, _find_adapter, _swap_attention_backend

PairLike = Tuple[str, str]


@dataclass
class _AttachConfig:
    """Stored on ``pipe._attn_vis_config`` after :func:`attach`."""
    words: Optional[Sequence[str]]
    attention_pairs: Optional[Sequence[PairLike]]
    save_dir: str
    heatmap_upscale: int
    fallback_to_sdpa: bool
    capture_chunk_size: int
    adapter: PipelineAdapter
    original_call: Any


def attach(
    pipe: Any,
    *,
    words: Optional[Sequence[str]] = None,
    attention_pairs: Optional[Sequence[PairLike]] = None,
    save_dir: str = "./attn_out",
    heatmap_upscale: int = 8,
    fallback_to_sdpa: bool = True,
    capture_chunk_size: int = 256,
    pipeline_type: Optional[str] = None,
) -> None:
    """Attach persistent attention capture to *pipe*.

    After this call every ``pipe(...)`` invocation will automatically
    capture joint-attention maps and save heatmaps / overlays to
    *save_dir*.  The pipeline return value is unchanged — ``pipe(...).images[0]``
    works exactly as before.

    Call :func:`detach` to remove the capture hook and restore the
    original pipeline behavior.

    Parameters
    ----------
    pipe
        A diffusion pipeline instance (e.g. ``Flux2KleinPipeline``).
    words
        Each word / phrase produces a spatial heatmap over the generated
        image showing where that phrase is attended.
    attention_pairs
        Low-level ``(query_selector, key_selector)`` tuples.  Can be
        combined with *words*.
    save_dir
        Output directory for heatmaps and overlays (created if missing).
    pipeline_type
        Explicit adapter name.  Auto-detected when ``None``.
    """
    pairs = _build_pair_list(words=words, attention_pairs=attention_pairs)
    if not pairs:
        raise ValueError(
            "attach() needs at least one target — pass words=[...] "
            "or attention_pairs=[...]."
        )

    if hasattr(pipe, "_attn_vis_config"):
        raise ValueError(
            "This pipeline already has attention capture attached. "
            "Call detach(pipe) first."
        )

    adapter = _find_adapter(pipe, pipeline_type)
    pipeline_cls = pipe.__class__
    original_call = pipeline_cls.__call__

    config = _AttachConfig(
        words=words,
        attention_pairs=attention_pairs,
        save_dir=save_dir,
        heatmap_upscale=heatmap_upscale,
        fallback_to_sdpa=fallback_to_sdpa,
        capture_chunk_size=capture_chunk_size,
        adapter=adapter,
        original_call=original_call,
    )
    pipe._attn_vis_config = config
    pipeline_cls.__call__ = _make_attached_call(original_call)


def detach(pipe: Any) -> None:
    """Remove attention capture from *pipe*, restoring the original ``__call__``."""
    config = getattr(pipe, "_attn_vis_config", None)
    if config is None:
        raise ValueError("Pipeline does not have attention capture attached.")
    pipe.__class__.__call__ = config.original_call
    del pipe._attn_vis_config


def _make_attached_call(original_call: Any) -> Any:
    """Build a wrapper ``__call__`` that captures attention when config is present."""

    @functools.wraps(original_call)
    def _attached_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        config: Optional[_AttachConfig] = getattr(self, "_attn_vis_config", None)
        if config is None:
            return original_call(self, *args, **kwargs)

        pairs = _build_pair_list(
            words=config.words,
            attention_pairs=config.attention_pairs,
        )
        state = CaptureState(
            save_dir=config.save_dir,
            attention_pairs=pairs,
            fallback_to_sdpa=config.fallback_to_sdpa,
            capture_chunk_size=config.capture_chunk_size,
            heatmap_upscale=config.heatmap_upscale,
        )
        state.reset()

        restore_backend = _swap_attention_backend()
        set_capture_state(state)
        try:
            result = config.adapter.patched_call(self, *args, **kwargs)
        finally:
            set_capture_state(None)
            restore_backend()

        output_images: List[Image.Image] = []
        if hasattr(result, "images") and isinstance(result.images, list):
            output_images = [
                img for img in result.images if isinstance(img, Image.Image)
            ]
        state.set_output_images(output_images)
        state.finalize()

        return result

    return _attached_call

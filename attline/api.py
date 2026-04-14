from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from .capture import CaptureState
from .patch import patched_pipeline

PairLike = Tuple[str, str]


def visualize_attention(
    *,
    pipe: Any,
    prompt: Union[str, List[str]],
    words: Optional[Sequence[str]] = None,
    attention_pairs: Optional[Sequence[PairLike]] = None,
    image: Optional[Union[Image.Image, Sequence[Image.Image]]] = None,
    save_dir: str = "./attn_out",
    heatmap_upscale: int = 8,
    fallback_to_sdpa: bool = True,
    capture_chunk_size: int = 256,
    pipeline_type: Optional[str] = None,
    **pipeline_kwargs: Any,
) -> Dict[str, Any]:
    """Run a diffusion pipeline call with joint-attention capture.

    Parameters
    ----------
    pipe
        A diffusion pipeline instance (e.g. ``Flux2KleinPipeline``).
        The pipeline type is auto-detected; pass *pipeline_type* to
        override.
    prompt
        The text prompt passed to the pipeline.
    words
        Optional convenience shortcut -- each word / phrase in the list is
        visualised as ``attention from the generated image to those text
        tokens``. Output files are named ``{word}_heatmap.png`` /
        ``{word}_overlay.png``.
    attention_pairs
        Optional low-level list of ``(query_selector, key_selector)`` tuples
        for full control. See *Selector forms* below. ``words`` and
        ``attention_pairs`` can be combined.
    image
        Optional condition image(s) for image-edit prompts.
    save_dir
        Output directory (created if missing).
    heatmap_upscale
        Integer upscale applied to the raw token-grid heatmap when it is saved
        to disk (default 8). Purely cosmetic — the captured attention is
        unchanged; the PNG is just larger and easier to read.
    fallback_to_sdpa
        On CUDA OOM during capture, fall back to ``F.scaled_dot_product_attention``
        for that call (loses capture for that call but lets generation finish).
    capture_chunk_size
        Query-chunk size for the streaming softmax used during capture. Lower
        values use less VRAM; higher values are faster.
    pipeline_type
        Explicit pipeline adapter name (e.g. ``"Flux2KleinPipeline"``). When
        ``None`` (default) the adapter is auto-detected from the pipeline
        class. Use ``supported_pipelines()`` to list available adapters.
    **pipeline_kwargs
        Forwarded to ``pipe(...)`` -- e.g. ``height``, ``width``,
        ``num_inference_steps``, ``guidance_scale``, ``generator``.

    Selector forms
    --------------
    - ``"text"`` / ``"noise"`` / ``"target"`` / ``"image[i]"`` -- full segments.
    - ``"text:'phrase'"`` -- the token range covering a phrase in the prompt
      (case-insensitive, first occurrence wins, subword-safe).
    - ``"image[i]:bbox(x0,y0,x1,y1)"`` -- reserved for a follow-up release.

    The heatmap is 2D when at least one side of a pair is spatial. When both
    are spatial, the key side is used (preserving v0.1.0 behavior). When only
    the query side is spatial (e.g. ``("noise", "text:'cat'")``) the heatmap
    is over the query -- a spatial map over the generated image showing
    *where* the phrase is attended.

    Returns
    -------
    dict with keys ``pipeline_output`` (the raw pipeline return value),
    ``saved_paths`` (``{label: {heatmap, overlay}}``), and
    ``skipped_capture_calls`` (number of attention calls that fell back to
    SDPA due to OOM; 0 in the happy path).
    """
    pairs = _build_pair_list(words=words, attention_pairs=attention_pairs)
    if not pairs:
        raise ValueError(
            "visualize_attention needs at least one pair -- pass `words=[...]` "
            "for the common case or `attention_pairs=[...]` for full control."
        )

    save_dir = str(Path(save_dir))
    if image is not None:
        if isinstance(image, Image.Image):
            image = [image]
        else:
            image = list(image)

    state = CaptureState(
        save_dir=save_dir,
        attention_pairs=pairs,
        fallback_to_sdpa=fallback_to_sdpa,
        capture_chunk_size=capture_chunk_size,
        heatmap_upscale=heatmap_upscale,
    )
    state.reset()

    with patched_pipeline(pipe, state, pipeline_type=pipeline_type):
        result = pipe(prompt=prompt, image=image, **pipeline_kwargs)

    output_images: List[Image.Image] = []
    if hasattr(result, "images") and isinstance(result.images, list):
        output_images = [img for img in result.images if isinstance(img, Image.Image)]
    state.set_output_images(output_images)
    saved_paths = state.finalize()

    return {
        "pipeline_output": result,
        "saved_paths": saved_paths,
        "skipped_capture_calls": state.skipped_capture_calls,
    }


def _build_pair_list(
    *,
    words: Optional[Sequence[str]],
    attention_pairs: Optional[Sequence[PairLike]],
) -> List[PairLike]:
    pairs: List[PairLike] = []
    seen: set = set()

    if words is not None:
        for w in words:
            if not isinstance(w, str) or not w.strip():
                raise ValueError(f"`words` entries must be non-empty strings; got {w!r}.")
            phrase = w.strip()
            pair = ("noise", f"text:'{phrase}'")
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)

    if attention_pairs is not None:
        for pair in attention_pairs:
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(tuple(pair))

    return pairs

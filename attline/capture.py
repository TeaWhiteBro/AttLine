from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .layouts import LayoutSpec
from .render import save_heatmap_image, save_overlay_image
from .selectors import Selection, TextMeta, resolve_selection, sanitize_name


_DEFAULT_KINDS = {"text", "noise", "image"}


def _stem_for_pair(q_sel: Selection, k_sel: Selection) -> str:
    """Pick a concise, informative filename stem for a resolved pair.

    When one side is a text phrase and the other is a full-range default
    segment (noise / text / image[i]), we just use the phrase — this makes
    the ``words=[...]`` shortcut produce ``hello_heatmap.png`` instead of
    ``noise_to_text_hello_heatmap.png``. Otherwise fall back to
    ``<q>_to_<k>``.
    """
    q_phrase = q_sel.phrase_text
    k_phrase = k_sel.phrase_text
    if q_phrase is not None and k_phrase is None and k_sel.kind in _DEFAULT_KINDS:
        return sanitize_name(q_phrase)
    if k_phrase is not None and q_phrase is None and q_sel.kind in _DEFAULT_KINDS:
        return sanitize_name(k_phrase)
    return f"{q_sel.sanitized}_to_{k_sel.sanitized}"


@dataclass
class SliceAccumulator:
    pair_key: Tuple[str, str]
    q_sel: Selection
    k_sel: Selection
    view_side: str
    stem: str = ""
    sum_map: Optional[torch.Tensor] = None
    count: int = 0

    # per-call scratch buffers (reset by state.begin_call)
    call_buf: Optional[torch.Tensor] = None
    call_weight: float = 0.0

    def begin_call(self) -> None:
        if self.view_side == "query":
            length = self.q_sel.length
        else:
            length = self.k_sel.length
        self.call_buf = torch.zeros(length, dtype=torch.float32)
        self.call_weight = 0.0

    def end_call(self) -> None:
        if self.call_buf is None:
            return
        if self.call_weight > 0:
            mean = self.call_buf / float(self.call_weight)
            if self.view_side == "query" and self.q_sel.shape_hw is not None:
                mean = mean.view(*self.q_sel.shape_hw)
            elif self.view_side == "key" and self.k_sel.shape_hw is not None:
                mean = mean.view(*self.k_sel.shape_hw)
            if self.sum_map is None:
                self.sum_map = torch.zeros_like(mean)
            self.sum_map += mean
            self.count += 1
        self.call_buf = None
        self.call_weight = 0.0

    def mean_map(self) -> torch.Tensor:
        if self.sum_map is None or self.count == 0:
            raise RuntimeError(f"No attention captured for pair={self.pair_key}.")
        return self.sum_map / float(self.count)


@dataclass
class CaptureState:
    save_dir: str
    attention_pairs: Sequence[Tuple[str, str]]
    fallback_to_sdpa: bool = True
    capture_chunk_size: int = 256
    heatmap_upscale: int = 8
    layout: Optional[LayoutSpec] = None
    text_meta: Optional[TextMeta] = None
    accumulators: Dict[Tuple[str, str], SliceAccumulator] = field(default_factory=dict)
    condition_images: List[Image.Image] = field(default_factory=list)
    output_images: List[Image.Image] = field(default_factory=list)
    skipped_capture_calls: int = 0

    def reset(self) -> None:
        self.layout = None
        self.text_meta = None
        self.accumulators = {}
        self.condition_images = []
        self.output_images = []
        self.skipped_capture_calls = 0

    def set_text_meta(self, text_meta: Optional[TextMeta]) -> None:
        self.text_meta = text_meta

    def set_layout(self, layout: LayoutSpec) -> None:
        self.layout = layout
        if self.text_meta is not None:
            text_seg = layout.get("text")
            self.text_meta.text_segment_start = text_seg.start
            self.text_meta.text_segment_end = text_seg.end

        self.accumulators = {}
        for pair in self.attention_pairs:
            q_name, k_name = pair
            q_sel = resolve_selection(q_name, layout, self.text_meta)
            k_sel = resolve_selection(k_name, layout, self.text_meta)
            view_side = _pick_view_side(q_sel, k_sel)
            self.accumulators[(q_name, k_name)] = SliceAccumulator(
                pair_key=(q_name, k_name),
                q_sel=q_sel,
                k_sel=k_sel,
                view_side=view_side,
                stem=_stem_for_pair(q_sel, k_sel),
            )

    def set_condition_images(self, images: Sequence[Image.Image]) -> None:
        self.condition_images = [img.copy() for img in images]

    def set_output_images(self, images: Sequence[Image.Image]) -> None:
        self.output_images = [img.copy() for img in images]

    def begin_call(self) -> None:
        for acc in self.accumulators.values():
            acc.begin_call()

    def end_call(self) -> None:
        for acc in self.accumulators.values():
            acc.end_call()

    def accumulate_chunk(
        self,
        attn_chunk: torch.Tensor,
        q_start: int,
        q_end: int,
    ) -> None:
        """Accumulate contribution from a query-chunked attention slice.

        `attn_chunk` has shape [B, H, q_end - q_start, K_total]. q indices in
        selections are absolute over the full joint sequence.
        """
        if self.layout is None:
            raise RuntimeError("CaptureState.layout is missing.")

        bh = attn_chunk.shape[0] * attn_chunk.shape[1]

        for acc in self.accumulators.values():
            q_sel = acc.q_sel
            k_sel = acc.k_sel

            q_rel, buf_pos = _chunk_q_overlap(q_sel, q_start, q_end)
            if q_rel is None:
                continue

            if q_rel is _FULL_CONTIG:
                rel_s, rel_e = buf_pos[0], buf_pos[1]
                bs, be = buf_pos[2], buf_pos[3]
                q_part = attn_chunk[:, :, rel_s:rel_e, :]
                buf_slice = (bs, be, None)
            else:
                q_idx = q_rel.to(attn_chunk.device)
                q_part = attn_chunk.index_select(2, q_idx)
                buf_slice = (None, None, buf_pos)

            if k_sel.contiguous_range is not None:
                ks, ke = k_sel.contiguous_range
                part = q_part[:, :, :, ks:ke]
            else:
                k_idx = k_sel.indices.to(attn_chunk.device)
                part = q_part.index_select(3, k_idx)
            # part shape: [B, H, q_part_len, k_len]

            if acc.view_side == "key":
                # sum over (batch, head, query), output shape [k_len]
                contrib = part.sum(dim=(0, 1, 2)).detach().to("cpu", dtype=torch.float32)
                acc.call_buf += contrib
                acc.call_weight += float(bh * part.shape[2])
            else:  # "query"
                # sum over (batch, head, k_selection), output shape [q_part_len]
                contrib = part.sum(dim=-1).sum(dim=(0, 1)).detach().to("cpu", dtype=torch.float32)
                bs, be, scatter_positions = buf_slice
                if scatter_positions is None:
                    acc.call_buf[bs:be] += contrib
                else:
                    acc.call_buf.index_add_(
                        0, scatter_positions.to(torch.long), contrib
                    )
                # weight tracks denominator per position; for query view every
                # position is hit exactly once per call, so track call_weight
                # as B*H (applied uniformly when we divide).
                if acc.call_weight == 0.0:
                    acc.call_weight = float(bh)

    def _base_image_for_selection(
        self, sel: Selection, view_side: str
    ) -> Optional[Image.Image]:
        if not sel.is_spatial:
            return None
        if sel.kind == "image":
            seg = self.layout.get(sel.name) if self.layout is not None else None
            if seg is not None and seg.source_index is not None:
                if 0 <= seg.source_index < len(self.condition_images):
                    return self.condition_images[seg.source_index]
        if sel.kind == "noise" and self.output_images:
            return self.output_images[0]
        return None

    def finalize(self) -> Dict[str, Dict[str, str]]:
        save_root = Path(self.save_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Dict[str, str]] = {}

        for pair_key, acc in self.accumulators.items():
            q_name, k_name = pair_key
            if acc.sum_map is None or acc.count == 0:
                continue
            mean_map = acc.mean_map().numpy()

            stem = acc.stem or f"{acc.q_sel.sanitized}_to_{acc.k_sel.sanitized}"
            heatmap_path = save_root / f"{stem}_heatmap.png"
            save_heatmap_image(mean_map, heatmap_path, upscale=self.heatmap_upscale)

            pair_out: Dict[str, str] = {"heatmap": str(heatmap_path)}
            spatial_sel = acc.q_sel if acc.view_side == "query" else acc.k_sel
            if spatial_sel.is_spatial:
                base = self._base_image_for_selection(spatial_sel, acc.view_side)
                if base is not None:
                    overlay_path = save_root / f"{stem}_overlay.png"
                    save_overlay_image(base, mean_map, overlay_path, upscale=self.heatmap_upscale)
                    pair_out["overlay"] = str(overlay_path)
            out[f"{q_name}->{k_name}"] = pair_out

        if not out and self.skipped_capture_calls > 0:
            raise RuntimeError(
                "Attention capture was skipped for all calls after fallback to SDPA; no maps were saved. "
                f"skipped_capture_calls={self.skipped_capture_calls}"
            )
        return out


def _pick_view_side(q_sel: Selection, k_sel: Selection) -> str:
    if q_sel.is_spatial and k_sel.is_spatial:
        return "key"
    if k_sel.is_spatial:
        return "key"
    if q_sel.is_spatial:
        return "query"
    return "key"


_FULL_CONTIG = object()  # sentinel: chunk overlap is a contiguous range


def _chunk_q_overlap(
    q_sel: Selection,
    q_start: int,
    q_end: int,
):
    """Compute the part of `q_sel` that falls inside [q_start, q_end).

    Returns `(None, None)` if there is no overlap.
    For a contiguous q_sel, returns `(_FULL_CONTIG, (rel_s, rel_e, buf_s, buf_e))`
    where `rel_*` index into the chunk tensor and `buf_*` index into the call
    buffer.
    Otherwise returns `(rel_idx_tensor, buf_pos_tensor)`.
    """
    if q_sel.contiguous_range is not None:
        qs, qe = q_sel.contiguous_range
        overlap_s = max(qs, q_start)
        overlap_e = min(qe, q_end)
        if overlap_s >= overlap_e:
            return None, None
        rel_s = overlap_s - q_start
        rel_e = overlap_e - q_start
        buf_s = overlap_s - qs
        buf_e = overlap_e - qs
        return _FULL_CONTIG, (rel_s, rel_e, buf_s, buf_e)

    mask = (q_sel.indices >= q_start) & (q_sel.indices < q_end)
    if not bool(mask.any()):
        return None, None
    buf_positions = torch.nonzero(mask, as_tuple=True)[0]
    abs_idxs = q_sel.indices[mask]
    rel_idxs = abs_idxs - q_start
    return rel_idxs, buf_positions


_CAPTURE_STATE: Optional[CaptureState] = None


def set_capture_state(state: Optional[CaptureState]) -> None:
    global _CAPTURE_STATE
    _CAPTURE_STATE = state


def get_capture_state() -> Optional[CaptureState]:
    return _CAPTURE_STATE


def _broadcast_mask_for_chunk(
    attn_mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    q: torch.Tensor,
    k: torch.Tensor,
) -> Optional[torch.Tensor]:
    if attn_mask is None:
        return None

    mask = attn_mask
    if mask.ndim == 2 and mask.shape[0] == q.shape[0] and mask.shape[1] == k.shape[2]:
        mask = mask.unsqueeze(1).unsqueeze(1)

    if mask.ndim == 4 and mask.shape[-2] == q.shape[2]:
        return mask[..., q_start:q_end, :]
    if mask.ndim == 4 and mask.shape[-2] == 1:
        return mask
    return mask


def _apply_gqa_if_needed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    enable_gqa: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enable_gqa:
        return query, key, value

    q_heads = query.shape[1]
    k_heads = key.shape[1]
    if q_heads == k_heads:
        return query, key, value
    if q_heads % k_heads != 0:
        raise ValueError(f"Cannot expand GQA heads: q_heads={q_heads}, k_heads={k_heads}")
    factor = q_heads // k_heads
    key = key.repeat_interleave(factor, dim=1)
    value = value.repeat_interleave(factor, dim=1)
    return query, key, value


def compute_attention_with_capture(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
    state: CaptureState,
) -> torch.Tensor:
    query, key, value = _apply_gqa_if_needed(query, key, value, enable_gqa)
    scale_value = float(scale) if scale is not None else (query.shape[-1] ** -0.5)
    q_len = query.shape[2]
    outputs = []

    state.begin_call()
    try:
        for q_start in range(0, q_len, state.capture_chunk_size):
            q_end = min(q_len, q_start + state.capture_chunk_size)
            q_chunk = query[:, :, q_start:q_end, :]
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale_value
            mask_chunk = _broadcast_mask_for_chunk(attn_mask, q_start, q_end, query, key)
            if mask_chunk is not None:
                if mask_chunk.dtype == torch.bool:
                    scores = scores.masked_fill(~mask_chunk, float("-inf"))
                else:
                    scores = scores + mask_chunk.to(device=scores.device, dtype=scores.dtype)

            if is_causal:
                causal_mask = torch.ones(
                    (q_end - q_start, key.shape[2]),
                    device=scores.device,
                    dtype=torch.bool,
                )
                causal_mask = torch.tril(causal_mask, diagonal=key.shape[2] - q_end + q_start)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            if dropout_p and dropout_p > 0:
                attn = torch.dropout(attn, dropout_p, train=True)

            state.accumulate_chunk(attn, q_start=q_start, q_end=q_end)
            out_chunk = torch.matmul(attn, value)
            outputs.append(out_chunk)
    finally:
        state.end_call()

    return torch.cat(outputs, dim=2)

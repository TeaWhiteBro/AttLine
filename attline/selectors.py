from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch

from .layouts import LayoutSpec, SegmentSpec, _normalize_segment_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    name: str
    sanitized: str
    indices: torch.Tensor
    shape_hw: Optional[Tuple[int, int]]
    kind: str
    contiguous_range: Optional[Tuple[int, int]] = None
    phrase_text: Optional[str] = None

    @property
    def length(self) -> int:
        return int(self.indices.numel())

    @property
    def is_spatial(self) -> bool:
        return self.shape_hw is not None


@dataclass
class TextMeta:
    chat_text: str
    offsets: List[Tuple[int, int]]
    text_segment_start: int = 0
    text_segment_end: int = 0


def sanitize_name(name: str) -> str:
    out = re.sub(r"[\[\]\"'():,\s/\\]+", "_", name.strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "sel"


def _range_selection(
    name: str,
    seg: SegmentSpec,
    kind: str,
    *,
    sub_start: Optional[int] = None,
    sub_end: Optional[int] = None,
    shape_hw: Optional[Tuple[int, int]] = None,
) -> Selection:
    start = seg.start if sub_start is None else sub_start
    end = seg.end if sub_end is None else sub_end
    indices = torch.arange(start, end, dtype=torch.long)
    if shape_hw is None and sub_start is None and sub_end is None:
        shape_hw = seg.shape_hw
    return Selection(
        name=name,
        sanitized=sanitize_name(name),
        indices=indices,
        shape_hw=shape_hw,
        kind=kind,
        contiguous_range=(start, end),
    )


_TEXT_SUB_RE = re.compile(r"""^text\s*:\s*(['"])(?P<phrase>.+)\1\s*$""")
_IMAGE_BBOX_RE = re.compile(
    r"""^image\s*\[\s*(?P<idx>\d+)\s*\]\s*:\s*bbox\s*\(\s*"""
    r"""(?P<x0>[-\d.]+)\s*,\s*(?P<y0>[-\d.]+)\s*,\s*"""
    r"""(?P<x1>[-\d.]+)\s*,\s*(?P<y1>[-\d.]+)\s*\)\s*$"""
)


def resolve_selection(
    name: str,
    layout: LayoutSpec,
    text_meta: Optional[TextMeta] = None,
) -> Selection:
    raw = name.strip()

    bbox_match = _IMAGE_BBOX_RE.match(raw)
    if bbox_match is not None:
        raise NotImplementedError(
            "image bbox selectors land in the next iteration; "
            f"parsed spec: {bbox_match.groupdict()}"
        )

    text_match = _TEXT_SUB_RE.match(raw)
    if text_match is not None:
        phrase = text_match.group("phrase")
        return _resolve_text_phrase(raw, phrase, layout, text_meta)

    normalized = _normalize_segment_name(raw)
    try:
        seg = layout.get(normalized)
    except KeyError as exc:
        raise KeyError(
            f"Unknown selector: {name!r}. Known segments: {layout.names()}. "
            f"Text sub-ranges use the form \"text:'phrase'\"."
        ) from exc

    return _range_selection(raw, seg, kind=seg.kind)


def _resolve_text_phrase(
    raw_name: str,
    phrase: str,
    layout: LayoutSpec,
    text_meta: Optional[TextMeta],
) -> Selection:
    if text_meta is None:
        raise ValueError(
            f"Cannot resolve {raw_name!r}: tokenizer metadata is unavailable. "
            "This usually means the pipeline call did not populate text_meta; "
            "ensure you are running inside patched_flux2_klein_pipeline."
        )

    chat_text = text_meta.chat_text
    haystack = chat_text.lower()
    needle = phrase.lower()
    if not needle:
        raise ValueError(f"Empty phrase in selector {raw_name!r}.")

    match_positions: List[int] = []
    start_search = 0
    while True:
        idx = haystack.find(needle, start_search)
        if idx == -1:
            break
        match_positions.append(idx)
        start_search = idx + 1

    if not match_positions:
        preview = chat_text[:120].replace("\n", "\\n")
        raise ValueError(
            f"Phrase {phrase!r} (from selector {raw_name!r}) not found in "
            f"chat-wrapped prompt. First ~120 chars of chat text: {preview!r}"
        )
    if len(match_positions) > 1:
        logger.warning(
            "Phrase %r appears %d times in prompt; using first match at char %d.",
            phrase,
            len(match_positions),
            match_positions[0],
        )

    char_start = match_positions[0]
    char_end = char_start + len(phrase)

    seg = layout.get("text")
    offsets = text_meta.offsets

    token_idxs: List[int] = []
    for local_idx, (c0, c1) in enumerate(offsets):
        if c0 == 0 and c1 == 0:
            continue
        if c1 <= char_start or c0 >= char_end:
            continue
        token_idxs.append(local_idx)

    if not token_idxs:
        raise ValueError(
            f"Phrase {phrase!r} matched in chat text at chars [{char_start},{char_end}) "
            "but no tokenizer offsets intersect it (unexpected)."
        )

    absolute_start = seg.start + token_idxs[0]
    absolute_end = seg.start + token_idxs[-1] + 1

    if absolute_end > seg.end:
        raise ValueError(
            f"Phrase {phrase!r} resolves to tokens outside the text segment "
            f"[{seg.start},{seg.end}); got [{absolute_start},{absolute_end})."
        )

    contiguous = token_idxs == list(range(token_idxs[0], token_idxs[-1] + 1))
    if contiguous:
        sel = _range_selection(
            raw_name,
            seg,
            kind="text_range",
            sub_start=absolute_start,
            sub_end=absolute_end,
            shape_hw=None,
        )
        return _with_phrase(sel, phrase)

    indices = torch.tensor(
        [seg.start + i for i in token_idxs], dtype=torch.long
    )
    return Selection(
        name=raw_name,
        sanitized=sanitize_name(raw_name),
        indices=indices,
        shape_hw=None,
        kind="text_range",
        contiguous_range=None,
        phrase_text=phrase,
    )


def _with_phrase(sel: Selection, phrase: str) -> Selection:
    return Selection(
        name=sel.name,
        sanitized=sel.sanitized,
        indices=sel.indices,
        shape_hw=sel.shape_hw,
        kind=sel.kind,
        contiguous_range=sel.contiguous_range,
        phrase_text=phrase,
    )


def has_text_phrase_selectors(pairs: Sequence[Tuple[str, str]]) -> bool:
    for q, k in pairs:
        for name in (q, k):
            if _TEXT_SUB_RE.match(name.strip()) is not None:
                return True
    return False


def compute_text_meta(
    tokenizer: Any,
    prompt: Any,
    max_sequence_length: int,
    *,
    use_chat_template: bool = True,
    template: Optional[str] = None,
    drop_prefix_tokens: int = 0,
    padding: str = "max_length",
) -> TextMeta:
    """Compute tokenizer offsets for text-phrase selection.

    Modes (mutually exclusive):
    - ``use_chat_template=True`` (default): wrap the prompt in a chat conversation
      and call ``tokenizer.apply_chat_template`` (Qwen2-VL / Flux2-klein style).
    - ``template=...``: a Python format string with one ``{}`` placeholder
      (QwenImage-style prompt_template_encode). ``drop_prefix_tokens`` strips
      the leading template tokens that the pipeline drops from ``prompt_embeds``.
    - Neither set: tokenize the raw prompt directly (Flux1 T5 style).
    """
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("Empty prompt list.")
        if len(prompt) > 1:
            logger.warning(
                "Text sub-range selectors currently only consider prompt[0]; "
                "got %d prompts.",
                len(prompt),
            )
        prompt_text = prompt[0]
    else:
        prompt_text = prompt

    if template is not None:
        chat_text = template.format(prompt_text)
    elif use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        conversation = [{"role": "user", "content": prompt_text}]
        chat_text = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        chat_text = prompt_text

    encoded = tokenizer(
        chat_text,
        padding=padding,
        truncation=True,
        max_length=max_sequence_length,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    raw_offsets = encoded["offset_mapping"]
    offsets: List[Tuple[int, int]] = [
        (int(a), int(b)) for a, b in raw_offsets
    ]

    if drop_prefix_tokens > 0:
        offsets = offsets[drop_prefix_tokens:]

    return TextMeta(chat_text=chat_text, offsets=offsets)

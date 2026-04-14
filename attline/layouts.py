from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    kind: str
    start: int
    end: int
    shape_hw: Optional[Tuple[int, int]] = None
    source_index: Optional[int] = None

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_spatial(self) -> bool:
        return self.shape_hw is not None


@dataclass(frozen=True)
class LayoutSpec:
    text_count: int
    segments: List[SegmentSpec]

    @property
    def total_tokens(self) -> int:
        return self.segments[-1].end if self.segments else self.text_count

    def get(self, name: str) -> SegmentSpec:
        normalized = _normalize_segment_name(name)
        for seg in self.segments:
            if seg.name == normalized:
                return seg
        raise KeyError(f"Unknown segment: {name}. Known segments: {[s.name for s in self.segments]}")

    def names(self) -> List[str]:
        return [seg.name for seg in self.segments]


def _normalize_segment_name(name: str) -> str:
    name = name.strip().lower()
    if name == "target":
        return "noise"
    return name


def normalize_to_multiple(width: int, height: int, multiple_of: int) -> Tuple[int, int]:
    width = max(multiple_of, (width // multiple_of) * multiple_of)
    height = max(multiple_of, (height // multiple_of) * multiple_of)
    return width, height


def compute_token_hw(height: int, width: int, downsample_factor: int) -> Tuple[int, int]:
    if height % downsample_factor != 0 or width % downsample_factor != 0:
        raise ValueError(
            f"Image size {(width, height)} is not divisible by downsample_factor={downsample_factor}."
        )
    return height // downsample_factor, width // downsample_factor


def build_flux2_klein_layout(
    *,
    text_count: int,
    noise_hw: Tuple[int, int],
    image_hws: Sequence[Tuple[int, int]],
) -> LayoutSpec:
    segments: List[SegmentSpec] = []
    cursor = 0

    segments.append(
        SegmentSpec(
            name="text",
            kind="text",
            start=cursor,
            end=cursor + text_count,
            shape_hw=None,
            source_index=None,
        )
    )
    cursor += text_count

    noise_tokens = noise_hw[0] * noise_hw[1]
    segments.append(
        SegmentSpec(
            name="noise",
            kind="noise",
            start=cursor,
            end=cursor + noise_tokens,
            shape_hw=noise_hw,
            source_index=None,
        )
    )
    cursor += noise_tokens

    for idx, hw in enumerate(image_hws):
        count = hw[0] * hw[1]
        segments.append(
            SegmentSpec(
                name=f"image[{idx}]",
                kind="image",
                start=cursor,
                end=cursor + count,
                shape_hw=hw,
                source_index=idx,
            )
        )
        cursor += count

    return LayoutSpec(text_count=text_count, segments=segments)
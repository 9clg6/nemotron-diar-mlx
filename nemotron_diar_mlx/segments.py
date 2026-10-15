"""Speaker probabilities -> speech segments.

Port of NeMo's `ts_vad_post_processing` (`binarization` + `filtering` from
`nemo.collections.asr.parts.utils.vad_utils`), used by
`SortformerEncLabelModel.diarize`. With default parameters the output matches
NeMo's segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: int

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "speaker": self.speaker}


def speaker_segments(
    preds: np.ndarray,
    frame_duration: float,
    onset: float = 0.5,
    offset: float = 0.5,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
) -> List[Segment]:
    """Segments of every speaker, sorted by start time.

    Args:
        preds: `(frames, num_speakers)` activity probabilities.
        frame_duration: seconds per frame of `preds`.
        onset / offset: hysteresis thresholds to start / end a segment.
        pad_onset / pad_offset: seconds added before / after each segment.
        min_duration_on: drop segments shorter than this (seconds).
        min_duration_off: fill gaps shorter than this (seconds).
    """
    segments = []
    for spk in range(preds.shape[1]):
        segs = binarize(preds[:, spk], frame_duration, onset, offset, pad_onset, pad_offset)
        segs = filter_segments(segs, min_duration_on, min_duration_off)
        segments.extend(Segment(float(s), float(e), spk) for s, e in segs)
    return sorted(segments, key=lambda s: (s.start, s.end, s.speaker))


def binarize(
    sequence: np.ndarray,
    frame_duration: float,
    onset: float = 0.5,
    offset: float = 0.5,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
) -> np.ndarray:
    """NeMo `binarization`: `(n, 2)` array of `[start, end]` seconds."""
    speech, start, i = False, 0.0, 0
    out = []
    for i, value in enumerate(sequence):
        if speech:
            if value < offset:
                if i * frame_duration + pad_offset > max(0.0, start - pad_onset):
                    out.append((max(0.0, start - pad_onset), i * frame_duration + pad_offset))
                start = i * frame_duration
                speech = False
        elif value > onset:
            start = i * frame_duration
            speech = True
    if speech:
        out.append((max(0.0, start - pad_onset), i * frame_duration + pad_offset))
    return _merge_overlaps(np.array(out, dtype=np.float64).reshape(-1, 2))


def filter_segments(segments: np.ndarray, min_duration_on: float = 0.0, min_duration_off: float = 0.0) -> np.ndarray:
    """NeMo `filtering` (short speech removed first, then short gaps filled)."""
    if segments.size == 0:
        return segments
    if min_duration_on > 0.0:
        segments = segments[segments[:, 1] - segments[:, 0] >= min_duration_on]
    if min_duration_off > 0.0 and len(segments) > 1:
        ordered = segments[np.argsort(segments[:, 0], kind="stable")]
        gaps = np.column_stack((ordered[:-1, 1], ordered[1:, 0]))
        short_gaps = gaps[gaps[:, 1] - gaps[:, 0] < min_duration_off]
        segments = _merge_overlaps(np.concatenate([segments, short_gaps]))
    return segments


def _merge_overlaps(segments: np.ndarray) -> np.ndarray:
    if len(segments) <= 1:
        return segments
    segments = segments[np.argsort(segments[:, 0], kind="stable")]
    merge = segments[:-1, 1] >= segments[1:, 0]
    heads = segments[~np.concatenate([[False], merge]), 0]
    tails = segments[~np.concatenate([merge, [False]]), 1]
    return np.stack([heads, tails], axis=1)

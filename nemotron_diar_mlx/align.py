"""Who said what: attribute time-stamped words (from any ASR) to speakers.

Each word goes to the speaker with the highest mean activity over the word's
span. Words with no clearly active speaker (e.g. a short pause mid-sentence)
inherit the speaker of the neighbouring words. Consecutive words of one
speaker are merged into turns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass
class Turn:
    speaker: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "start": round(self.start, 2), "end": round(self.end, 2), "text": self.text}


def speaker_for_span(
    preds: np.ndarray, frame_duration: float, start: float, end: float, min_activity: float = 0.3
) -> Optional[int]:
    """Most active speaker over `[start, end)`, or None if none reaches `min_activity`."""
    if preds.shape[0] == 0:
        return None
    a = min(max(int(start / frame_duration), 0), preds.shape[0] - 1)
    b = min(max(math.ceil(end / frame_duration), a + 1), preds.shape[0])
    activity = preds[a:b].mean(axis=0)
    best = int(np.argmax(activity))
    return best if activity[best] >= min_activity else None


def assign_speakers(
    words: Sequence[Word], preds: np.ndarray, frame_duration: float, min_activity: float = 0.3
) -> List[Optional[int]]:
    """Speaker of each word; undecided words take the previous (else next) decided speaker."""
    speakers = [speaker_for_span(preds, frame_duration, w.start, w.end, min_activity) for w in words]
    last = None
    for i, spk in enumerate(speakers):
        if spk is None:
            speakers[i] = last
        else:
            last = spk
    nxt = None
    for i in range(len(speakers) - 1, -1, -1):
        if speakers[i] is None:
            speakers[i] = nxt
        else:
            nxt = speakers[i]
    return speakers


def words_to_turns(
    words: Iterable[Word],
    preds: np.ndarray,
    frame_duration: float,
    min_activity: float = 0.3,
    renumber: bool = True,
) -> List[Turn]:
    """Group words into speaker turns.

    Word texts are concatenated as-is (keep leading spaces to mark word
    boundaries, as SentencePiece-based ASR such as Parakeet produces). With
    `renumber`, speakers are renumbered 0, 1, 2... by first appearance among
    the turns (the model already numbers them by arrival, but a speaker may
    have no words).
    """
    words = [w for w in words if w.text.strip()]
    speakers = assign_speakers(words, preds, frame_duration, min_activity)
    turns: List[Turn] = []
    for word, spk in zip(words, speakers):
        spk = -1 if spk is None else spk  # no speech detected at all
        if turns and turns[-1].speaker == spk:
            turns[-1].end = word.end
            turns[-1].text += word.text
        else:
            turns.append(Turn(spk, word.start, word.end, word.text))
    for turn in turns:
        turn.text = " ".join(turn.text.split())
    if renumber:
        mapping: dict = {}
        for turn in turns:
            if turn.speaker >= 0:
                turn.speaker = mapping.setdefault(turn.speaker, len(mapping))
    return turns


def tokens_to_words(tokens: Iterable) -> List[Word]:
    """Merge sub-word tokens (objects with `text`, `start`, `end`) into words.

    A token whose text starts with a space begins a new word, which is how
    SentencePiece tokenizers (Parakeet, Canary...) mark word boundaries.
    """
    words: List[Word] = []
    for tok in tokens:
        if not words or tok.text.startswith(" "):
            words.append(Word(tok.start, tok.end, tok.text))
        else:
            last = words[-1]
            words[-1] = Word(last.start, max(last.end, tok.end), last.text + tok.text)
    return words

from types import SimpleNamespace

import numpy as np

from nemotron_diar_mlx.align import Word, assign_speakers, speaker_for_span, tokens_to_words, words_to_turns


def _preds():
    preds = np.zeros((300, 4))
    preds[0:100, 2] = 0.95  # speaker 2 speaks first (0-1 s)
    preds[120:200, 0] = 0.9  # then speaker 0 (1.2-2 s)
    preds[180:300, 1] = 0.8  # speaker 1 overlaps at the end
    return preds


def test_speaker_for_span():
    preds = _preds()
    assert speaker_for_span(preds, 0.01, 0.1, 0.5) == 2
    assert speaker_for_span(preds, 0.01, 1.3, 1.6) == 0
    assert speaker_for_span(preds, 0.01, 1.0, 1.15) is None  # silence
    assert speaker_for_span(preds, 0.01, 5.0, 6.0) == 1  # past the end: last frame


def test_undecided_words_inherit_neighbours():
    words = [Word(0.0, 0.2, " a"), Word(1.0, 1.15, " b"), Word(1.3, 1.5, " c")]
    assert assign_speakers(words, _preds(), 0.01) == [2, 2, 0]
    assert assign_speakers([Word(1.0, 1.15, " x"), Word(1.3, 1.5, " y")], _preds(), 0.01) == [0, 0]


def test_turns_are_merged_and_renumbered():
    words = [Word(0.0, 0.3, " Hello"), Word(0.3, 0.6, " there."), Word(1.3, 1.5, " Hi!"), Word(2.2, 2.6, " Yes")]
    turns = words_to_turns(words, _preds(), 0.01)
    assert [(t.speaker, t.text) for t in turns] == [(0, "Hello there."), (1, "Hi!"), (2, "Yes")]
    assert turns[0].to_dict() == {"speaker": 0, "start": 0.0, "end": 0.6, "text": "Hello there."}
    raw = words_to_turns(words, _preds(), 0.01, renumber=False)
    assert [t.speaker for t in raw] == [2, 0, 1]


def test_no_speech_at_all_gives_unknown_speaker():
    turns = words_to_turns([Word(0, 1, " hm")], np.zeros((100, 4)), 0.01)
    assert [(t.speaker, t.text) for t in turns] == [(-1, "hm")]


def test_tokens_to_words():
    toks = [SimpleNamespace(text=t, start=s, end=e) for t, s, e in
            ((" Bon", 0.0, 0.2), ("jour", 0.2, 0.4), (" à", 0.5, 0.6), (" tous", 0.6, 0.9), (".", 0.9, 0.95))]
    assert tokens_to_words(toks) == [Word(0.0, 0.4, " Bonjour"), Word(0.5, 0.6, " à"), Word(0.6, 0.95, " tous.")]

"""MLX (Apple Silicon) port of NVIDIA Nemotron 3 Diarization."""

from .align import Turn, Word, tokens_to_words, words_to_turns
from .config import STREAMING_PRESETS, ModelConfig
from .model import Model
from .segments import Segment, speaker_segments
from .streaming import StreamingDiarizer, StreamingOutput

__version__ = "0.1.0"
__all__ = [
    "Model",
    "ModelConfig",
    "STREAMING_PRESETS",
    "Segment",
    "StreamingDiarizer",
    "StreamingOutput",
    "Turn",
    "Word",
    "speaker_segments",
    "tokens_to_words",
    "words_to_turns",
]

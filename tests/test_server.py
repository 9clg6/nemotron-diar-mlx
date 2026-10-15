import asyncio
import json
import urllib.request

import numpy as np
import pytest

pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

from nemotron_diar_mlx.server import DiarizationServer  # noqa: E402

from .conftest import random_model, tiny_config, voice_like_audio  # noqa: E402


async def _run_session(port: int, pcm: bytes, block: int):
    frames = {}
    async with connect(f"ws://127.0.0.1:{port}/v1/diarize") as ws:
        hello = json.loads(await ws.recv())
        for i in range(0, len(pcm), block):
            await ws.send(pcm[i : i + block])
        await ws.send(json.dumps({"type": "flush"}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "frames":
                for k, row in enumerate(msg["probs"]):
                    frames[msg["start_frame"] + k] = row
            elif msg["type"] == "done":
                return hello, frames, msg
            else:
                raise AssertionError(msg)


def test_server_streams_the_same_frames_as_direct_inference():
    model = random_model(tiny_config())
    audio = voice_like_audio(6.0)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    expected = model(pcm.astype(np.float32) / 32768.0)["preds"]
    server = DiarizationServer(model)

    async def main():
        async with serve(server.handle, "127.0.0.1", 0, process_request=server.process_request) as srv:
            port = srv.sockets[0].getsockname()[1]
            url = f"http://127.0.0.1:{port}/health"
            # Not ready yet: health and sessions are refused.
            with pytest.raises(urllib.error.HTTPError):
                await asyncio.to_thread(urllib.request.urlopen, url)
            server.ready = True
            assert (await asyncio.to_thread(urllib.request.urlopen, url)).status == 200
            # Odd block sizes: samples split across messages.
            return await _run_session(port, pcm.tobytes(), block=3001)

    hello, frames, done = asyncio.run(main())
    assert hello["type"] == "ready" and hello["num_speakers"] == 4
    assert done["num_frames"] == len(expected) == len(frames)
    got = np.array([frames[i] for i in range(len(frames))])
    np.testing.assert_allclose(got, expected, atol=2e-3)  # server rounds to 3 decimals


def test_warm_up_runs():
    DiarizationServer(random_model(tiny_config())).warm_up(3.0)

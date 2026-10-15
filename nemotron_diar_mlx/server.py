"""Live diarization over WebSocket.

    python -m nemotron_diar_mlx.server [--port 8001] [--preset low_latency] [--parent-pid PID]

Protocol (one diarization session per connection, endpoint `/v1/diarize`):

- client -> server: **binary** messages carrying raw PCM, 16-bit little-endian,
  mono, 16 kHz, in blocks of any size; then the **text** message
  `{"type": "flush"}` to finalize the session.
- server -> client (JSON text messages):
  - `{"type": "ready", "frame_duration": 0.01, "num_speakers": 8, "latency": 1.04}` on connect;
  - `{"type": "frames", "start_frame": n, "probs": [[p0, ..., p7], ...]}` whenever new
    frames are final: probability that each speaker is active, one row per
    `frame_duration` seconds, `start_frame` counted from the first sample sent;
  - `{"type": "done", "num_frames": n}` after a flush.

`GET /health` answers 200 once the model is loaded and warmed up, 503 before.
Speakers are numbered by order of first appearance, as in the offline model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from http import HTTPStatus
from typing import Optional

import mlx.core as mx
import numpy as np

from .config import STREAMING_PRESETS
from .convert import DEFAULT_REPO
from .model import Model
from .streaming import StreamingDiarizer, StreamingOutput

log = logging.getLogger("nemotron_diar_mlx.server")


class DiarizationServer:
    def __init__(self, model: Model):
        self.model = model
        self.ready = False
        # MLX evaluation is serialized: one chunk at a time across connections.
        self._compute = asyncio.Lock()

    # -- lifecycle

    def warm_up(self, seconds: float) -> None:
        """Run synthetic audio through a session so every tensor shape is compiled before real use.

        The cache and FIFO grow during the first minute of a session, so each
        early chunk has a new shape; compiling them up front avoids stalls on
        the first live session.
        """
        if seconds <= 0:
            return
        rng = np.random.default_rng(0)
        t = np.arange(int(seconds * 16000)) / 16000
        audio = 0.01 * rng.standard_normal(t.shape)
        for i, f0 in enumerate((110.0, 170.0, 230.0)):
            gate = (np.sin(2 * np.pi * (0.13 + 0.05 * i) * t + i) > 0.3).astype(np.float64)
            audio += 0.2 * gate * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5))
        diar = StreamingDiarizer(self.model)
        for i in range(0, len(audio), 8000):
            diar.feed(audio[i : i + 8000].astype(np.float32))
        diar.flush()

    # -- HTTP

    def process_request(self, connection, request):
        if request.path == "/health":
            if self.ready:
                return connection.respond(HTTPStatus.OK, "ok\n")
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "loading\n")
        if not self.ready:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "loading\n")
        if request.path.split("?")[0] not in ("/v1/diarize", "/"):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None

    # -- WebSocket session

    async def handle(self, websocket) -> None:
        diar = StreamingDiarizer(self.model)
        cfg = self.model.config
        await websocket.send(json.dumps({
            "type": "ready",
            "frame_duration": self.model.frame_duration,
            "num_speakers": cfg.modules.num_spks,
            "latency": round(diar.latency, 3),
        }))
        pending: list = []
        wake = asyncio.Event()
        finished = asyncio.Event()

        async def worker() -> None:
            try:
                await process()
            except Exception as exc:  # report instead of leaving the client waiting
                log.exception("diarization failed")
                await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
            finally:
                finished.set()

        async def process() -> None:
            while True:
                await wake.wait()
                wake.clear()
                while pending:
                    # Merge what queued up while the previous chunk was computing.
                    items, pending[:] = list(pending), []
                    audio = [x for x in items if isinstance(x, np.ndarray)]
                    flush = any(x is None for x in items)
                    if audio:
                        block = np.concatenate(audio)
                        async with self._compute:
                            out = await asyncio.to_thread(diar.feed, block)
                        await self._send_frames(websocket, out)
                    if flush:
                        async with self._compute:
                            out = await asyncio.to_thread(diar.flush)
                        await self._send_frames(websocket, out)
                        await websocket.send(json.dumps({"type": "done", "num_frames": out.start_frame + len(out.preds)}))
                        return

        task = asyncio.create_task(worker())
        carry = b""  # odd trailing byte: a sample split across two messages
        try:
            async for message in websocket:
                if finished.is_set():
                    break
                if isinstance(message, (bytes, bytearray)):
                    data = carry + bytes(message)
                    usable = len(data) - len(data) % 2
                    carry = data[usable:]
                    if usable:
                        pending.append(np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0)
                        wake.set()
                else:
                    try:
                        msg = json.loads(message)
                    except ValueError:
                        await websocket.send(json.dumps({"type": "error", "message": "invalid JSON"}))
                        continue
                    if msg.get("type") == "flush":
                        pending.append(None)
                        wake.set()
                        await finished.wait()
                        break
                    await websocket.send(json.dumps({"type": "error", "message": f"unknown type {msg.get('type')!r}"}))
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    async def _send_frames(websocket, out: StreamingOutput) -> None:
        if len(out.preds) == 0:
            return
        probs = np.round(out.preds.astype(np.float64), 3).tolist()
        await websocket.send(json.dumps({"type": "frames", "start_frame": out.start_frame, "probs": probs}))


def _watch_parent(pid: int) -> None:
    """Exit when the launching process dies (avoids orphan servers holding the port)."""

    def run() -> None:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                log.info("parent %s exited, stopping", pid)
                os._exit(0)
            except PermissionError:
                pass
            time.sleep(2.0)

    threading.Thread(target=run, daemon=True, name="parent-watchdog").start()


async def _serve(args) -> None:
    from websockets.asyncio.server import serve

    t0 = time.time()
    model = Model.from_pretrained(args.model, dtype=getattr(mx, args.dtype), streaming=args.preset,
                                  cache_dir=args.cache_dir)
    server = DiarizationServer(model)
    async with serve(server.handle, args.host, args.port, process_request=server.process_request,
                     max_size=None, compression=None):
        log.info("listening on %s:%d, warming up", args.host, args.port)
        await asyncio.to_thread(server.warm_up, args.warmup)
        server.ready = True
        print(f"NEMOTRON_DIAR_READY port={args.port} preset={args.preset} "
              f"load={time.time() - t0:.1f}s", flush=True)
        await asyncio.Future()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m nemotron_diar_mlx.server", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default=DEFAULT_REPO)
    parser.add_argument("--preset", default="low_latency", choices=sorted(STREAMING_PRESETS))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--warmup", type=float, default=60.0, help="seconds of synthetic audio to pre-compile shapes")
    parser.add_argument("--parent-pid", type=int, help="exit when this process dies")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    if args.parent_pid:
        _watch_parent(args.parent_pid)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


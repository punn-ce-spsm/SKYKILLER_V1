"""Where `Detection` messages go.

Build 1 writes JSONL. Phase P2 stands up the broker and flips `sink.mqtt.enabled`;
the lane code does not change, because it only ever calls `emit`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from .config import Config
from .schemas import Detection


class Sink(Protocol):
    def emit(self, det: Detection) -> None: ...
    def close(self) -> None: ...


class StdoutSink:
    """One JSON object per line on stdout, so the lane pipes into anything."""

    def emit(self, det: Detection) -> None:
        # Flush per line: a consumer piping this is watching a live sensor, and
        # block buffering would hold detections back until the buffer fills.
        sys.stdout.write(det.to_json() + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.flush()


class JsonlSink:
    """Append to a file. Also builds the corpus for fine-tuning later."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def emit(self, det: Detection) -> None:
        self._fh.write(det.to_json() + "\n")

    def close(self) -> None:
        self._fh.close()


class MqttSink:
    """Publish onto the detection bus. paho-mqtt is imported only if used."""

    def __init__(self, host: str, port: int, topic: str) -> None:
        import paho.mqtt.client as mqtt  # noqa: PLC0415 -- optional dependency

        self._topic = topic
        self._client = mqtt.Client()
        self._client.connect(host, port, keepalive=60)
        self._client.loop_start()

    def emit(self, det: Detection) -> None:
        self._client.publish(self._topic, det.to_json())

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


class MultiSink:
    """Fan out to several sinks. A failing sink must not stop the lane."""

    def __init__(self, sinks: list[Sink]) -> None:
        self._sinks = sinks

    def emit(self, det: Detection) -> None:
        for s in self._sinks:
            try:
                s.emit(det)
            except Exception as exc:  # noqa: BLE001 -- a dead sink is not a dead sensor
                print(f"[sink] {type(s).__name__} failed: {exc}", file=sys.stderr)

    def close(self) -> None:
        for s in self._sinks:
            try:
                s.close()
            except Exception:  # noqa: BLE001, S110
                pass


def build(cfg: Config) -> MultiSink:
    sinks: list[Sink] = []
    if cfg.sink.stdout:
        sinks.append(StdoutSink())
    if cfg.sink.jsonl_path:
        sinks.append(JsonlSink(cfg.sink.jsonl_path))
    if cfg.sink.mqtt.enabled:
        sinks.append(MqttSink(cfg.sink.mqtt.host, cfg.sink.mqtt.port, cfg.sink.mqtt.topic))
    return MultiSink(sinks)

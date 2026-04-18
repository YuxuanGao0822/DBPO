"""Minimal video recorder used by DBPO env runners."""

from __future__ import annotations

import av
import numpy as np


class VideoRecorder:
    """Small PyAV-backed recorder for env-rendered RGB videos."""

    def __init__(self, fps: int, codec: str, input_pix_fmt: str, **kwargs) -> None:
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        self._reset_state()

    def _reset_state(self) -> None:
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None

    @classmethod
    def create_h264(
        cls,
        fps: int,
        codec: str = "h264",
        input_pix_fmt: str = "rgb24",
        output_pix_fmt: str = "yuv420p",
        crf: int = 18,
        profile: str = "high",
        **kwargs,
    ) -> "VideoRecorder":
        return cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile": profile},
            **kwargs,
        )

    def __del__(self) -> None:
        self.stop()

    def is_ready(self) -> bool:
        return self.stream is not None

    def start(self, file_path: str) -> None:
        if self.is_ready():
            self.stop()
        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for key, value in self.kwargs.items():
            setattr(codec_context, key, value)

    def write_frame(self, img: np.ndarray) -> None:
        if not self.is_ready():
            raise RuntimeError("VideoRecorder.start() must be called before write_frame().")
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            height, width, _ = img.shape
            self.stream.width = width
            self.stream.height = height
        assert img.shape == self.shape
        assert img.dtype == self.dtype
        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def stop(self) -> None:
        if not self.is_ready():
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._reset_state()

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shared.onnx import create_onnx_session


class FixedShapeOnnxModel:
    """ONNX model backed by stable Torch buffers and CUDA I/O binding."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        device: torch.device,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.device = device
        self.cuda_stream_ptr = (
            None if cuda_stream is None else int(cuda_stream.cuda_stream)
        )
        self.input = torch.zeros(input_shape, dtype=torch.float32, device=device)
        self.output = torch.empty(output_shape, dtype=torch.float32, device=device)
        self.session = create_onnx_session(
            model_path,
            device=str(device),
            cuda_stream=cuda_stream,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._validate_signature(input_shape, output_shape)
        self._binding = self._create_cuda_binding() if device.type == "cuda" else None

    def run(self) -> torch.Tensor:
        if self._binding is not None:
            self.session.run_with_iobinding(self._binding)
        else:
            result = self.session.run(
                [self.output_name], {self.input_name: self.input.numpy()}
            )[0]
            self.output.copy_(torch.from_numpy(np.asarray(result, dtype=np.float32)))
        return self.output

    def _create_cuda_binding(self):
        binding = self.session.io_binding()
        binding.bind_input(
            self.input_name,
            "cuda",
            0,
            np.float32,
            tuple(self.input.shape),
            self.input.data_ptr(),
        )
        binding.bind_output(
            self.output_name,
            "cuda",
            0,
            np.float32,
            tuple(self.output.shape),
            self.output.data_ptr(),
        )
        return binding

    def _validate_signature(
        self,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
    ) -> None:
        actual_input = tuple(self.session.get_inputs()[0].shape)
        actual_output = tuple(self.session.get_outputs()[0].shape)
        if actual_input != input_shape:
            raise RuntimeError(
                f"Unexpected ONNX input shape {actual_input}; expected {input_shape}"
            )
        if actual_output != output_shape:
            raise RuntimeError(
                f"Unexpected ONNX output shape {actual_output}; expected {output_shape}"
            )

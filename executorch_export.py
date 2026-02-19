"""
uv venv
source .venv/bin/activate
uv pip install executorch -e '.[dev]' 

Example:

python executorch_export.py \
  --model-id nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --output-path exports/cosmos_transformer.pte

Reference:
https://docs.pytorch.org/executorch/stable/using-executorch-export.html
"""

# from __future__ import annotations # NOTE bug for msup (to patch) field is not resolved into a type

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.export import Dim, export

from diffusers import Cosmos2_5_PredictBasePipeline
from msup.cli import cli


@dataclass
class ExportModelArgs:
    model_id: str = "nvidia/Cosmos-Predict2.5-2B"
    revision: str = "diffusers/base/post-trained"
    output_path: str = "exports/cosmos_transformer.pte"
    batch_size: int = 1
    num_frames: int = 17
    height: int = 64
    width: int = 64
    text_seq_len: int = 512
    use_dynamic_shapes: bool = False
    use_xnnpack: bool = True
    run_after_export: bool = True
    compare_original: bool = False


class ExportableCosmosTransformer(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]


def _get_text_embed_dim(transformer: torch.nn.Module) -> int:
    if transformer.config.use_crossattn_projection:
        return int(transformer.config.crossattn_proj_in_channels)
    return int(transformer.config.text_embed_dim)


def _build_example_inputs(
    transformer: torch.nn.Module, args: ExportModelArgs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = next(transformer.parameters()).dtype
    # Cosmos forward concatenates `condition_mask` into channels before patch embedding,
    # so latent input channels are one less than `config.in_channels`.
    in_channels = int(transformer.config.in_channels) - 1
    text_embed_dim = _get_text_embed_dim(transformer)

    hidden_states = torch.randn(
        args.batch_size,
        in_channels,
        args.num_frames,
        args.height,
        args.width,
        dtype=dtype,
    )
    timestep = torch.rand(args.batch_size, 1, args.num_frames, 1, 1, dtype=dtype)
    encoder_hidden_states = torch.randn(args.batch_size, args.text_seq_len, text_embed_dim, dtype=dtype)
    condition_mask = torch.rand(args.batch_size, 1, args.num_frames, args.height, args.width, dtype=dtype)
    padding_mask = torch.zeros(1, 1, args.height, args.width, dtype=dtype)

    return hidden_states, timestep, encoder_hidden_states, condition_mask, padding_mask

def run_executorch_model(
    output_path: Path,
    example_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> List[torch.Tensor]:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(str(output_path))
    try:
        method = program.load_method("forward")
    except RuntimeError as error:
        error_message = str(error)
        if "XnnpackBackend" in error_message or "Failed to load method forward" in error_message:
            raise RuntimeError(
                "Failed to load ExecuTorch method `forward`. "
                "The exported model likely uses XNNPACK delegation that is not supported in this runtime. "
                "Re-export with `use_xnnpack=false` and try again."
            ) from error
        raise
    return method.execute(list(example_inputs))


def _run_and_compare_with_eager(
    eager_reference_model: torch.nn.Module,
    output_path: Path,
    example_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    with torch.no_grad():
        eager_output = eager_reference_model(*example_inputs)

    executorch_output = run_executorch_model(output_path, example_inputs)
    is_close = torch.allclose(executorch_output[0], eager_output, rtol=1e-3, atol=1e-5)
    print("Run successfully via executorch runtime")
    print("Comparing against original PyTorch module")
    print(is_close)
    return bool(is_close)


def _torch_dtype_from_scalar_type_code(scalar_type_code: int) -> torch.dtype:
    # c10::ScalarType enum values used by ExecuTorch TensorInfo.dtype()
    mapping = {
        0: torch.uint8,
        1: torch.int8,
        2: torch.int16,
        3: torch.int32,
        4: torch.int64,
        5: torch.float16,
        6: torch.float32,
        7: torch.float64,
        11: torch.bool,
        15: torch.bfloat16,
    }
    if scalar_type_code not in mapping:
        raise ValueError(f"Unsupported ExecuTorch scalar type code: {scalar_type_code}")
    return mapping[scalar_type_code]


def _make_tensor_from_tensor_info(tensor_info: Any) -> torch.Tensor:
    shape = tuple(int(dim) for dim in tensor_info.sizes())
    dtype = _torch_dtype_from_scalar_type_code(int(tensor_info.dtype()))
    if dtype in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        return torch.randn(shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype)


def _build_example_inputs_from_export(
    output_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(str(output_path))
    method_meta = program.metadata("forward")

    if method_meta.num_inputs() != 5:
        raise ValueError(f"Expected 5 inputs for `forward`, got {method_meta.num_inputs()}")

    inputs = tuple(
        _make_tensor_from_tensor_info(method_meta.input_tensor_meta(index))
        for index in range(method_meta.num_inputs())
    )
    return inputs  # type: ignore[return-value]


def _build_dynamic_shapes(transformer: torch.nn.Module, args: ExportModelArgs) -> Dict[str, Any]:
    max_size = getattr(transformer.config, "max_size", (args.num_frames, args.height, args.width))
    max_height = max(args.height, int(max_size[1]))
    max_width = max(args.width, int(max_size[2]))

    h_dim = Dim("h", min=args.height, max=max_height)
    w_dim = Dim("w", min=args.width, max=max_width)

    return {
        "hidden_states": {3: h_dim, 4: w_dim},
        "timestep": None,
        "encoder_hidden_states": None,
        "condition_mask": {3: h_dim, 4: w_dim},
        "padding_mask": {2: h_dim, 3: w_dim},
    }


def export_model(
    args: ExportModelArgs,
):
    from executorch.exir import to_edge_transform_and_lower

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_id, revision=args.revision, torch_dtype=torch.float32
    )
    transformer = pipe.transformer.to(device="cpu", dtype=torch.float32).eval()
    del pipe

    wrapped_model = ExportableCosmosTransformer(transformer).eval()
    example_inputs = _build_example_inputs(transformer, args)
    dynamic_shapes = _build_dynamic_shapes(transformer, args) if args.use_dynamic_shapes else None

    with torch.no_grad():
        exported_program = export(wrapped_model, args=example_inputs, dynamic_shapes=dynamic_shapes)

    if args.use_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        try:
            edge_manager = to_edge_transform_and_lower(
                exported_program,
                partitioner=[XnnpackPartitioner()],
            )
        except Exception as error:
            print(
                f"XNNPACK partitioning failed ({error.__class__.__name__}: {error}). "
                "Falling back to unpartitioned ExecuTorch export."
            )
            edge_manager = to_edge_transform_and_lower(exported_program)
    else:
        edge_manager = to_edge_transform_and_lower(exported_program)

    executorch_program = edge_manager.to_executorch()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(executorch_program.buffer)
    print(f"Exported ExecuTorch model to {output_path}")

    if args.run_after_export:
        _run_and_compare_with_eager(
            eager_reference_model=wrapped_model,
            output_path=output_path,
            example_inputs=example_inputs,
        )


def test_model(args: ExportModelArgs):
    output_path = Path(args.output_path)
    if not output_path.exists():
        raise FileNotFoundError(
            f"ExecuTorch model not found at {output_path}. Run `export_model` first."
        )

    example_inputs = _build_example_inputs_from_export(output_path)
    output = run_executorch_model(output_path, example_inputs)
    print("Run successfully via executorch runtime")
    print(f"Num outputs: {len(output)}")
    if output:
        print(f"Output[0] shape: {tuple(output[0].shape)}")
        print(f"Output[0] dtype: {output[0].dtype}")

    if args.compare_original:
        pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
            args.model_id, revision=args.revision, torch_dtype=torch.float32
        )
        transformer = pipe.transformer.to(device="cpu", dtype=torch.float32).eval()
        del pipe

        wrapped_model = ExportableCosmosTransformer(transformer).eval()
        _run_and_compare_with_eager(
            eager_reference_model=wrapped_model,
            output_path=output_path,
            example_inputs=example_inputs,
        )

if __name__ == "__main__":
    cli({
        export_model: "exports a model",
        test_model: "tests a model that has been exported"
    })

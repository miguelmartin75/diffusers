"""
Example:
python executorch_export.py \
  --model-id nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --output-path exports/cosmos_transformer.pte

Reference:
https://docs.pytorch.org/executorch/stable/using-executorch-export.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.export import Dim, export

from diffusers import Cosmos2_5_PredictBasePipeline


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


def export_model(args: ExportModelArgs) -> Path:
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
    return output_path


def _parse_args() -> ExportModelArgs:
    parser = argparse.ArgumentParser(description="Export Cosmos transformer to ExecuTorch (.pte).")
    parser.add_argument("--model-id", default=ExportModelArgs.model_id)
    parser.add_argument("--revision", default=ExportModelArgs.revision)
    parser.add_argument("--output-path", default=ExportModelArgs.output_path)
    parser.add_argument("--batch-size", type=int, default=ExportModelArgs.batch_size)
    parser.add_argument("--num-frames", type=int, default=ExportModelArgs.num_frames)
    parser.add_argument("--height", type=int, default=ExportModelArgs.height)
    parser.add_argument("--width", type=int, default=ExportModelArgs.width)
    parser.add_argument("--text-seq-len", type=int, default=ExportModelArgs.text_seq_len)
    parser.add_argument("--dynamic-shapes", action="store_true")
    parser.add_argument("--disable-xnnpack", action="store_true")
    parsed = parser.parse_args()

    return ExportModelArgs(
        model_id=parsed.model_id,
        revision=parsed.revision,
        output_path=parsed.output_path,
        batch_size=parsed.batch_size,
        num_frames=parsed.num_frames,
        height=parsed.height,
        width=parsed.width,
        text_seq_len=parsed.text_seq_len,
        use_dynamic_shapes=parsed.dynamic_shapes,
        use_xnnpack=not parsed.disable_xnnpack,
    )


def main() -> None:
    output_path = export_model(_parse_args())
    print(f"Exported ExecuTorch model to {output_path}")


if __name__ == "__main__":
    main()

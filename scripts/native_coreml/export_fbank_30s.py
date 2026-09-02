#!/usr/bin/env python3
"""Export a fixed-30s-window fbank CoreML model (wespeaker-fbank-30s.mlmodelc).

Mirrors export_b64_seg.py's pattern (fixed shape, not EnumeratedShapes) and
convert_coreml.py's export_fbank() conversion parameters, for the one asset
neither existing script produces. Required by speakrs' native CoreML chunk
pipeline (src/inference/embedding/native/loaders.rs:
  model_path.with_file_name("wespeaker-fbank-30s.mlmodelc")
), fixed shape [1, 1, 480_000] (30s @ 16kHz) per
src/inference/embedding/load/sessions.rs's CachedInputShape::new("waveform", &[1, 1, 480_000]).
"""

import argparse
from pathlib import Path
import coremltools as ct
import numpy as np
import torch
from common import (
    build_fbank_wrapper,
    save_model_artifacts,
    coreml_packages_dir,
)


def deployment_target():
    return ct.target.macOS13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the fixed-30s-window fbank CoreML model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/models"),
        help="Directory where the compiled .mlmodelc bundle should be written",
    )
    return parser.parse_args()


def main():
    output_dir = parse_args().output_dir
    fbank_wrapper = build_fbank_wrapper()

    # Fixed batch=1, 480_000 samples (30s @ 16kHz) -- matches the Rust side's
    # fbank_30s_buf / CachedInputShape exactly. Same rationale as export_b64_seg.py:
    # a fixed shape, not EnumeratedShapes, for this one large chunk-boundary case.
    example = torch.zeros(1, 1, 480_000, dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(fbank_wrapper, example)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="waveform",
                shape=(1, 1, 480_000),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=deployment_target(),
        compute_precision=ct.precision.FLOAT32,
    )

    stem = "wespeaker-fbank-30s"
    compiled = [output_dir / f"{stem}.mlmodelc"]
    pkg = coreml_packages_dir(output_dir) / f"{stem}.mlpackage"

    print(f"Saving {stem} (FP32)...")
    save_model_artifacts(mlmodel, pkg, compiled)
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
probe_migraphx_compile_hang.py
==============================

Bisect the MIGraphXExecutionProvider lazy-compile hang
(program::compile -> repeat_while_changes pass loop never converging).

Every configuration is run under a hard `timeout` so a hang can never stall
this script — a hanging config is reported as TIMEOUT and the script moves on.

Usage
-----
    # Against the REAL model that hangs (recommended):
    timeout 900 python3 scripts/probe_migraphx_compile_hang.py --model model.onnx

    # Sanity-check the harness with a small synthetic model:
    python3 scripts/probe_migraphx_compile_hang.py

    # Tune per-run timeout / skip slow levers:
    python3 scripts/probe_migraphx_compile_hang.py --model model.onnx --timeout 180 --skip L8

The script builds ONE InferenceSession per lever and forces the first
sess.run() (the point where MIGraphX EP compiles lazily). Session build time
is reported separately so you can see the 6s-build / hang-on-run split.

Output: a summary table plus a recommended working configuration.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np

REQUIRED_MODULES = ("onnxruntime", "onnx", "numpy")
for _m in REQUIRED_MODULES:
    try:
        __import__(_m)
    except ImportError:
        sys.exit(f"missing module: {_m} (run: python3 -m pip install {_m})")

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

# ---------------------------------------------------------------------------
# Timeout enforcement (works on POSIX; Windows uses the shell `timeout` tool)
# ---------------------------------------------------------------------------


class TimeoutExpired(Exception):
    pass


class _Alarm:
    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return False

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutExpired("compile/run exceeded timeout")


def run_under_timeout(fn, seconds: float, label: str):
    """Run `fn`, aborting with TimeoutExpired after `seconds`."""
    if seconds <= 0:
        return fn()
    with _Alarm(seconds):
        return fn()


# ---------------------------------------------------------------------------
# Synthetic model (used only when --model is not given): dynamic-batch graph
# with reshape/gather/slice patterns of the kind that stress MIGraphX's
# reshape-simplification passes. Real-world repro should use --model.
# ---------------------------------------------------------------------------


def build_synthetic_model(batch: int, seq: int, hidden: int, dynamic_batch: bool) -> bytes:
    X = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None if dynamic_batch else batch, seq, hidden]
    )
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None if dynamic_batch else batch, hidden])

    # Cast -> Gather(axis=1, indices=[0]) -> Squeeze -> MatMul(W) -> Add(b) -> Relu
    w = helper.make_tensor("W", TensorProto.FLOAT, [hidden, hidden], np.random.randn(hidden, hidden).astype(np.float32).flatten())
    b = helper.make_tensor("B", TensorProto.FLOAT, [hidden], np.random.randn(hidden).astype(np.float32).flatten())
    idx = helper.make_tensor("IDX", TensorProto.INT64, [1], [0])
    # Opset 13 removed Squeeze's `axes` attribute; it is now an optional
    # second INPUT tensor (emitting `axes=[1]` as an attribute fails ORT
    # model load with INVALID_GRAPH).
    axes = helper.make_tensor("AXES", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Gather", ["input", "IDX"], ["gathered"], axis=1),
        helper.make_node("Squeeze", ["gathered", "AXES"], ["squeezed"]),
        helper.make_node("MatMul", ["squeezed", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["out"]),
        helper.make_node("Relu", ["out"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "synthetic_dynamic", [X], [Y], initializer=[w, b, idx, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# Lever definitions
# ---------------------------------------------------------------------------


@dataclass
class Lever:
    id: str
    desc: str
    providers: list
    graph_opt: object
    provider_opts: dict
    optimize_first: bool = False
    note: str = ""


def make_levers(model_bytes: bytes, batch: int, seq: int, hidden: int, dynamic_batch: bool) -> list[Lever]:
    migx = "MIGraphXExecutionProvider"
    cpu = "CPUExecutionProvider"

    return [
        Lever(
            id="L0",
            desc="CPU-only (baseline unblock)",
            providers=[cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={},
            note="No MIGraphX compile at all — always works, slowest.",
        ),
        Lever(
            id="L1",
            desc="MIGraphX + ORT_ENABLE_ALL (REPRO)",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={},
            note="Expected to hang on the dynamic graph — this is the bug.",
        ),
        Lever(
            id="L2",
            desc="MIGraphX + ORT_DISABLE_ALL",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            provider_opts={},
            note="ORT hands the EP an unoptimized subgraph — different reshape pattern.",
        ),
        Lever(
            id="L3",
            desc="MIGraphX + ORT_ENABLE_BASIC",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            provider_opts={},
        ),
        Lever(
            id="L4",
            desc="MIGraphX + exhaustive_tune=0",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={"migraphx_exhaustive_tune": "0"},
            note="Env equivalent: export ORT_MIGRAPHX_EXHAUSTIVE_TUNE=0",
        ),
        Lever(
            id="L5",
            desc="MIGraphX + fp16_enable=0",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={"migraphx_fp16_enable": "0"},
            note="Env equivalent: export ORT_MIGRAPHX_FP16_ENABLE=0 (rusty-stack already sets this)",
        ),
        Lever(
            id="L6",
            desc="MIGraphX + STATIC batch shape",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={},
            note="Fixed dims remove the dynamic-dimension pass work.",
        ),
        Lever(
            id="L7",
            desc="MIGraphX + SMALL batch/seq",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={},
            note="Smaller shapes change which reshape patterns appear.",
        ),
        Lever(
            id="L8",
            desc="ORT pre-optimized model -> MIGraphX",
            providers=[migx, cpu],
            graph_opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            provider_opts={},
            optimize_first=True,
            note="Offline ORT_ENABLE_ALL transform, then EP compiles the optimized graph.",
        ),
    ]


# ---------------------------------------------------------------------------
# Per-lever run
# ---------------------------------------------------------------------------


def run_lever(model_path: str, lever: Lever, timeout: float, batch: int, seq: int, hidden: int):
    """Return (status, build_s, run_s, detail). status in PASS/TIMEOUT/ERROR."""
    import tempfile

    model_to_load = model_path
    if lever.optimize_first:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp.close()
        model_to_load = tmp.name
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = model_to_load
        try:
            # CPU-only: an EP must NOT run during pre-opt or the optimized
            # graph bakes in EP-compiled nodes that cannot be re-serialized.
            ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            return "ERROR", 0.0, 0.0, f"pre-optimize failed: {exc}"

    try:
        build_t0 = time.monotonic()

        def _build():
            opts = ort.SessionOptions()
            opts.graph_optimization_level = lever.graph_opt
            opts.log_severity_level = 3
            kw = dict(sess_options=opts, providers=[(p, lever.provider_opts) if p != "CPUExecutionProvider" else p for p in lever.providers])
            return ort.InferenceSession(model_to_load, **kw)

        session = run_under_timeout(_build, timeout, "session build")
        build_s = time.monotonic() - build_t0
        print(f"    [{lever.id}] session built in {build_s:.1f}s; providers={session.get_providers()}", flush=True)

        # Feed EVERY model input with a type-appropriate tensor. Multi-input
        # models (e.g. input_ids + attention_mask) fail with "required inputs
        # missing" if only the first input is fed.
        feed = {}
        for inp in session.get_inputs():
            dims = [d if isinstance(d, int) else batch for d in inp.shape]
            t = inp.type
            if "int64" in t:
                feed[inp.name] = np.random.randint(0, 1024, size=dims).astype(np.int64)
            elif "int32" in t:
                feed[inp.name] = np.random.randint(0, 1024, size=dims).astype(np.int32)
            elif "uint8" in t:
                feed[inp.name] = np.random.randint(0, 255, size=dims).astype(np.uint8)
            else:
                feed[inp.name] = np.random.randn(*dims).astype(np.float32)

        run_t0 = time.monotonic()
        run_under_timeout(lambda: session.run(None, feed), timeout, "first run")
        run_s = time.monotonic() - run_t0
        return "PASS", build_s, run_s, f"first run ok ({run_s:.1f}s)"

    except TimeoutExpired:
        return "TIMEOUT", 0.0, 0.0, f"hung > {timeout:.0f}s (compile loop not converging)"
    except Exception as exc:  # noqa: BLE001
        return "ERROR", 0.0, 0.0, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Bisect MIGraphX lazy-compile hangs")
    ap.add_argument("--model", default=None, help="Path to the ONNX model that hangs (default: synthetic)")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-lever compile timeout in seconds (default 120)")
    ap.add_argument("--skip", action="append", default=[], help="Lever id(s) to skip (repeatable)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    print("=" * 72)
    print("Environment")
    print("=" * 72)
    print(f"onnxruntime     : {ort.__version__}")
    print(f"providers       : {ort.get_available_providers()}")
    try:
        with open("/opt/rocm/.info/version") as fh:
            print(f"rocm            : {fh.read().strip()}")
    except OSError:
        print("rocm            : <not found at /opt/rocm>")
    for var in ("ORT_MIGRAPHX_FP16_ENABLE", "ORT_MIGRAPHX_EXHAUSTIVE_TUNE", "ORT_MIGRAPHX_MODEL_CACHE_PATH"):
        print(f"{var:<28}: {os.environ.get(var, '<unset>')}")

    model_path = args.model
    if model_path is None:
        # Build the three synthetic variants up front: the dynamic-batch repro
        # (default for all levers), a STATIC-shape twin (L6), and a SMALL
        # static model (L7). Each is saved once and reused across runs.
        dynamic_path = "/tmp/migraphx_probe_dynamic.onnx"
        static_path = "/tmp/migraphx_probe_static.onnx"
        small_path = "/tmp/migraphx_probe_small.onnx"
        onnx.save(
            onnx.load_model_from_string(
                build_synthetic_model(args.batch, args.seq, args.hidden, dynamic_batch=True)
            ),
            dynamic_path,
        )
        onnx.save(
            onnx.load_model_from_string(
                build_synthetic_model(args.batch, args.seq, args.hidden, dynamic_batch=False)
            ),
            static_path,
        )
        onnx.save(
            onnx.load_model_from_string(build_synthetic_model(1, 8, 32, dynamic_batch=False)),
            small_path,
        )
        print(f"Using synthetic models: dynamic={dynamic_path}, static={static_path}, small={small_path}")
    else:
        dynamic_path = static_path = small_path = model_path
        print(f"Using model     : {model_path}")

    levers = make_levers(model_path, args.batch, args.seq, args.hidden, dynamic_batch=(args.model is None))
    results = []
    for lever in levers:
        if lever.id in args.skip:
            print(f"\n[{lever.id}] SKIPPED ({lever.desc})")
            continue
        # L6 exercises a STATIC-batch model, L7 a SMALL static model; every
        # other lever shares the dynamic-batch repro model.
        lever_model = dynamic_path
        if lever.id == "L6":
            lever_model = static_path
        elif lever.id == "L7":
            lever_model = small_path
        print(f"\n[{lever.id}] {lever.desc} (model={lever_model})")
        status, build_s, run_s, detail = run_lever(lever_model, lever, args.timeout, args.batch, args.seq, args.hidden)
        print(f"    -> {status}: {detail}")
        results.append((lever.id, lever.desc, status, build_s, run_s))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'Lever':<5} {'Status':<9} {'Build(s)':<9} {'Run(s)':<9} Description")
    for lid, desc, status, build_s, run_s in results:
        print(f"{lid:<5} {status:<9} {build_s:<9.1f} {run_s:<9.1f} {desc}")

    passes = [r for r in results if r[2] == "PASS"]
    if passes:
        print("\nWORKING CONFIGURATIONS: " + ", ".join(r[0] for r in passes))
        print("Recommended: pick the fastest PASSING lever that keeps MIGraphX, "
              "then harden it (env var in ~/.mlstack_env, provider options in code).")
    else:
        print("\nNO LEVER WORKED within the timeout. Next step: MIGraphX library-level fix")
        print("(upgrade/patch MIGraphX 2.15.0, or use CPU-only in the interim).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

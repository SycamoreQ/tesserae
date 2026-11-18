import subprocess
from pathlib import Path

import modal


def ignore_local_artifacts(path: Path) -> bool:
    parts = path.parts
    return ".git" in parts or "_build" in parts or path.name == ".DS_Store"


tesserae_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ocaml", "opam", "git", "build-essential", "gcc-multilib")
    .run_commands(
        "opam init --disable-sandboxing -y",
        "eval $(opam env)",
        "opam install dune base stdio alcotest -y",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/nvidia/cutlass.git /tmp/cutlass_latest"
    )
    .add_local_dir(".", remote_path="/workspace", ignore=ignore_local_artifacts)
)

app = modal.App("tesserae-gpu-runner")

# Shared by every function.
BASE_ENV = {
    "C_INCLUDE_PATH": "/usr/include:/usr/include/x86_64-linux-gnu:/usr/local/cuda/include",
    "CPLUS_INCLUDE_PATH": "/usr/include:/usr/include/x86_64-linux-gnu:/usr/local/cuda/include",
    "CUTLASS_PATH": "/tmp/cutlass_latest/include",
}

# Tests: launches are synchronous, so an error is reported at the launch that
# caused it. NEVER use this for the benchmark: it serialises every launch.
TEST_ENV = {**BASE_ENV, "CUDA_LAUNCH_BLOCKING": "1"}


def _sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(f"eval $(opam env) && cd /workspace && {cmd}", shell=True, **kw)


def _build() -> bool:
    print("Building OCaml project...")
    r = _sh("dune build", capture_output=True, text=True)
    if r.returncode != 0:
        print("❌ Build failed:\n", r.stderr)
        return False
    return True


@app.function(image=tesserae_image, gpu="B200", timeout=600, env=TEST_ENV)
def run_tesserae_tests():
    if not _build():
        return
    print("Build successful! Running test suite...")
    r = _sh("timeout 300s dune test")
    print("All tests passed!" if r.returncode == 0
          else "Test suite failed. Check the logs above.")


@app.function(image=tesserae_image, gpu="B200", timeout=900, env=BASE_ENV)
def run_tesserae_bench(args: str = ""):
    """GEMM timing baseline. No CUDA_LAUNCH_BLOCKING here (see TEST_ENV)."""
    if not _build():
        return
    subprocess.run(
        "nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm --format=csv",
        shell=True,
    )
    print(f"Running benchmark: bench_gemm {args}")
    r = _sh(f"timeout 600s dune exec lib/runtime/bench/bench_gemm.exe -- {args}")
    if r.returncode != 0:
        print("Benchmark failed or timed out. Check the logs above.")


@app.function(image=tesserae_image, timeout=600, env=BASE_ENV)
def check_build():
    """No GPU needed, so it starts immediately even when B200s are queued.
    The link step may fail here if the image has no libcuda; in that case use
    the GPU functions instead."""
    if _build():
        print("Build OK.")


@app.local_entrypoint()
def main(bench: bool = False, build_only: bool = False, bench_args: str = ""):
    """
    modal run run_modal.py                      # tests
    modal run run_modal.py --bench              # benchmark, default 4096^3
    modal run run_modal.py --bench --bench-args "-m 2048 -n 2048 -k 2048"
    modal run run_modal.py --build-only         # compile check, no GPU
    """
    if build_only:
        check_build.remote()
    elif bench:
        run_tesserae_bench.remote(bench_args)
    else:
        run_tesserae_tests.remote()

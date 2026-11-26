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

torch_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .add_local_file("cublas_bench.py", remote_path="/workspace/cublas_bench.py")
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


@app.function(image=tesserae_image, gpu="B200", timeout=600, env=BASE_ENV)
def run_tesserae_ncu(args: str = ""):
    """Nsight Compute profile of one steady-state bench launch.

    Runs the compiled bench binary DIRECTLY (no `dune exec` under injection),
    with an explicit metric list so ncu needs a single replay pass, and a
    minimal bench run by default (1 warmup, 1 iter, 1 repeat) so the app
    exits right after the profiled launch. Sections/multi-pass collection is
    what made the first version appear to hang.
    """
    if not _build():
        return
    subprocess.run(
        "nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader", shell=True)
    r = _sh(
        'NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu '
        '/opt/nvidia/nsight-compute*/ncu 2>/dev/null | head -1); '
        'echo "ncu: ${NCU:-MISSING}"; "$NCU" --version | head -2',
        capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0 or "MISSING" in r.stdout:
        print("ncu not found in image:", r.stderr[-2000:])
        return
    if not args.strip():
        args = ("-m 4096 -n 4096 -k 4224 -stages 3 -bn 256 -pairs 64 "
                "-warmup 1 -iters 1 -repeats 1 -l2 128")
    # Discover the Blackwell tcgen05 tensor-pipe counter name (differs from the
    # Hopper hmma metric). Cheap, no profiling; printed so we can add it next.
    r = _sh(
        'NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu '
        '/opt/nvidia/nsight-compute*/ncu 2>/dev/null | head -1); '
        '"$NCU" --query-metrics 2>/dev/null '
        '| grep -iE "pipe_tensor|tcgen|tensor_op" | head -40',
        capture_output=True, text=True)
    print("tensor-pipe metric candidates:\n" + (r.stdout or "(none)")[:4000])
    metrics = ",".join([
        "gpu__time_duration.sum",
        "launch__registers_per_thread",
        "launch__waves_per_multiprocessor",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_warps",
        "launch__occupancy_limit_blocks",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "dram__bytes.sum",
        "lts__t_bytes.sum",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    ])
    exe = "_build/default/lib/runtime/bench/bench_gemm.exe"
    print(f"Running ncu on {exe} {args}")
    _sh(
        f'NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu '
        f'/opt/nvidia/nsight-compute*/ncu 2>/dev/null | head -1); '
        f'timeout 480s "$NCU" --kernel-name gemm_blackwell_ws '
        f'--launch-skip 1 --launch-count 1 --clock-control none '
        f'--metrics {metrics} {exe} {args}'
    )


@app.function(image=tesserae_image, timeout=600, env=BASE_ENV)
def check_build():
    """No GPU needed, so it starts immediately even when B200s are queued.
    The link step may fail here if the image has no libcuda; in that case use
    the GPU functions instead."""
    if _build():
        print("Build OK.")

@app.local_entrypoint()
def main(bench: bool = False, build_only: bool = False, dump: bool = False,
         test_head: bool = False, test_logs: bool = False, cublas: bool = False,
         ncu: bool = False, bench_args: str = "", cublas_args: str = "",
         ncu_args: str = ""):
    """
    modal run run_modal.py                      # tests
    modal run run_modal.py --bench              # benchmark, default 4096^3
    modal run run_modal.py --bench --bench-args "-m 2048 -n 2048 -k 2048"
    modal run run_modal.py --build-only         # compile check, no GPU
    modal run run_modal.py --ncu                # ncu profile of one launch
    modal run run_modal.py --ncu --ncu-args "-m 4096 -n 4096 -k 4096 -stages 3"
    modal run run_modal.py --cublas             # cuBLAS reference timing
    modal run run_modal.py --cublas --cublas-args "--shapes 4096x4096x4104"
    """
    if build_only:
        check_build.remote()
    elif test_head:
        run_tesserae_tests_head.remote()
    elif test_logs:
        cat_test_logs.remote()
    elif dump:
        dump_kernel.remote()
    elif ncu:
        run_tesserae_ncu.remote(ncu_args)
    elif cublas:
        run_cublas.remote(cublas_args)
    elif bench:
        run_tesserae_bench.remote(bench_args)
    else:
        run_tesserae_tests.remote()


@app.function(image=tesserae_image, timeout=900, env=TEST_ENV, gpu="B200")
def cat_test_logs():
    if not _build():
        return
    r = _sh("timeout 600s dune test lib/runtime 2>&1 | tail -5; "
            "for f in /workspace/_build/default/lib/runtime/tests/_build/_tests/Runtime/gemm_blackwell_ragged.*.output; do "
            "echo ===$f===; cat $f; done", capture_output=True, text=True)
    print(r.stdout[-8000:])


@app.function(image=tesserae_image, timeout=900, env=TEST_ENV, gpu="B200")
def run_tesserae_tests_head():
    """Runs dune test against the HEAD state (working-tree diff reversed)."""
    if not _build():
        return
    p = _sh("patch -R -p1 < .head_patch.diff", capture_output=True, text=True)
    if p.returncode != 0:
        print("reverse patch failed:", p.stderr[-500:])
        return
    r = _sh("timeout 600s dune test")
    print("HEAD tests:", "passed" if r.returncode == 0 else "FAILED")
    _sh("patch -p1 < .head_patch.diff")


@app.function(image=tesserae_image, timeout=600, env=BASE_ENV)
def dump_kernel():
    if not _build():
        return
    r = _sh("dune exec bin/dump_kernel.exe -- 4104 > /workspace/kernel_new.cu", capture_output=True, text=True)
    if r.returncode != 0:
        print("dump new failed:", r.stderr[-2000:])
        return
    p = _sh("patch -R -p1 < .head_patch.diff", capture_output=True, text=True)
    if p.returncode != 0:
        print("reverse patch failed:", p.stdout[-1000:], p.stderr[-1000:])
        return
    r = _sh("dune exec bin/dump_kernel.exe -- 4104 > /workspace/kernel_head.cu", capture_output=True, text=True)
    if r.returncode != 0:
        print("dump head failed:", r.stderr[-2000:])
        return
    p = _sh("patch -p1 < .head_patch.diff", capture_output=True, text=True)
    if p.returncode != 0:
        print("re-apply patch failed:", p.stderr[-1000:])
    import re
    def clean(path):
        t = open(path).read()
        # strip first "Generated kernel" marker duplication
        return t
    d = subprocess.run("diff /workspace/kernel_head.cu /workspace/kernel_new.cu", shell=True, capture_output=True, text=True)
    print(d.stdout)
    print(d.stderr)

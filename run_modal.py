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


@app.function(
    image=tesserae_image,
    gpu="H100",
    timeout=600,
    env={
        "C_INCLUDE_PATH": "/usr/include:/usr/include/x86_64-linux-gnu:/usr/local/cuda/include",
        "CPLUS_INCLUDE_PATH": "/usr/include:/usr/include/x86_64-linux-gnu:/usr/local/cuda/include",
        "CUTLASS_PATH": "/tmp/cutlass_latest/include",
        "CUDA_LAUNCH_BLOCKING": "1",
    },
)
def run_tesserae_tests():
    setup_cmd = "eval $(opam env) && cd /workspace && dune build"

    print("Building OCaml project on Linux H100 instance...")
    build_result = subprocess.run(setup_cmd, shell=True, capture_output=True, text=True)

    if build_result.returncode != 0:
        print("❌ Build failed:\n", build_result.stderr)
        return

    print("Build successful! Running test suite...")

    test_cmd = "eval $(opam env) && cd /workspace && timeout 120s dune test"
    test_result = subprocess.run(test_cmd, shell=True)

    if test_result.returncode == 0:
        print("All tests passed!")
    else:
        print("Test suite failed. Check the logs above.")

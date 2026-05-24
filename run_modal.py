import subprocess

import modal

# 1. Define the Linux environment with CUDA, OCaml, and required packages
tesserae_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ocaml", "opam", "git", "build-essential")
    .run_commands(
        "opam init --disable-sandboxing -y",
        "eval $(opam env)",
        # FIX 1: alcotest added to the installation list
        "opam install dune base stdio alcotest -y",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/nvidia/cutlass.git /tmp/cutlass_latest"
    )
    # Syncs your local Mac code to /workspace on the remote GPU
    .add_local_dir(".", remote_path="/workspace")
)

app = modal.App("tesserae-gpu-runner")


# FIX 2: Upgraded GPU to H100
@app.function(image=tesserae_image, gpu="H100", timeout=600)
def run_tesserae_tests():
    # Step 1: Build the project
    setup_cmd = "eval $(opam env) && cd /workspace && dune build"

    print("Building OCaml project on Linux H100 instance...")
    build_result = subprocess.run(setup_cmd, shell=True, capture_output=True, text=True)

    if build_result.returncode != 0:
        print("❌ Build failed:\n", build_result.stderr)
        return

    print("Build successful! Running test suite...")

    # Step 2: Run the tests with forced unbuffered output
    test_cmd = "eval $(opam env) && cd /workspace && dune test --force --no-buffer"
    test_result = subprocess.run(test_cmd, shell=True)

    if test_result.returncode == 0:
        print("All tests passed!")
    else:
        print("Test suite failed. Check the logs above.")

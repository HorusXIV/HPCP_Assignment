import subprocess
import sys
import os


def test_multigpu_entrypoint_importable():
    """Simple smoke test: ensure the multiGPU entrypoint can be executed in serial.

    This test runs the main module with a very small synthetic input to ensure
    imports and basic wiring do not error on CI/development machines.
    """
    # Allow overriding via env var so CI or local fixtures can provide a path.
    input_path = os.environ.get(
        "HPCP_TEST_INPUT", "data/np32/20170906_12_00_12.npz"
    )
    # Run the module with python -m to simulate the Slurm/launcher usage
    cmd = [sys.executable, "-m", "src.multiGPU.main", "--input", input_path]
    # Allow the command to run; it should not raise an exception. We don't
    # strictly assert on output because environment differences (MPI, cupy)
    # may change runtime logs.
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
    except subprocess.CalledProcessError:
        # If the process fails, tests should fail clearly.
        raise AssertionError("multiGPU.main failed")

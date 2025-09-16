import subprocess
import sys
import os


def test_multigpu_entrypoint_importable():
    """Simple smoke test: ensure the multiGPU entrypoint can be executed in serial.

    This test runs the main module with a very small synthetic input to ensure
    imports and basic wiring do not error on CI/development machines.
    """
    # Run the module with python -m to simulate the Slurm/launcher usage
    cmd = [sys.executable, "-m", "src.multiGPU.main", "--input", "data/np32/20170906_12_00_12.npz"]
    # Allow the command to run; it should not raise an exception. We don't
    # strictly assert on output because environment differences (MPI, cupy)
    # may change runtime logs.
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
    except subprocess.CalledProcessError as e:
        # If the process fails, include stdout/stderr in assertion message
        raise AssertionError(f"multiGPU.main failed: stdout={e.stdout}\nstderr={e.stderr}")

CuPy setup and verification
===========================

This project requires a CUDA-enabled CuPy wheel for multiGPU execution. The
`pyproject.toml` currently declares `cupy-cuda12x = "^13.6.0"` which will
install a CuPy wheel built for CUDA 12.x if your environment's package
index and system CUDA drivers are compatible.

Quick checklist
---------------

1. Verify system CUDA driver version (on Linux/Windows):

   - Windows: check `nvidia-smi` output for CUDA Version / Driver Version.
   - Linux: run `nvidia-smi`.

2. Match CuPy wheel to CUDA driver:

   - If your system has CUDA 12.x drivers, the `cupy-cuda12x` wheel is
     appropriate (example: `cupy-cuda12x==13.6.0`).
   - If you have CUDA 11.x drivers, use `cupy-cuda11x` matching the minor
     series (e.g. `cupy-cuda11x==13.6.0` if a build exists) or install
     via `pip` with a specific wheel from CuPy's index.

3. Install into your Poetry environment

   Poetry does not always pick binary wheels from alternate indexes; the
   recommended approach is:

   - Activate your Poetry shell:

     ```pwsh
     poetry shell
     ```

   - Install CuPy with pip inside the venv created by Poetry so the binary
     wheel is fetched and installed correctly:

     ```pwsh
     python -m pip install "cupy-cuda12x==13.6.0"
     ```

   - Alternatively, add `cupy-cuda12x = "^13.6.0"` to `pyproject.toml` and
     run `poetry update`, but if Poetry fails to locate a wheel you may
     need to use the pip install step above.

Verification
------------

After installing, verify CuPy + CUDA availability:

```pwsh
python -c "import cupy; print('cupy', cupy.__version__); print('cuda', cupy.cuda.runtime.getDeviceCount())"
```

Expected output: CuPy reports a version and `getDeviceCount()` returns >= 1.

Troubleshooting
---------------

- If `import cupy` raises "No module named 'cupy'": ensure you installed the
  wheel into the same Python environment your application uses (Poetry venv).

- If `cupy` imports but `cupy.cuda.runtime.getDeviceCount()` returns 0 or a
  CUDA error occurs, verify drivers are installed and that the GPU is
  visible to the user running the process. On clusters, ensure the job
  scheduler allocates GPU resources and that `CUDA_VISIBLE_DEVICES` is set.

- For locked-down clusters where binary wheels cannot be installed, consider
  using the system package manager or a container image with a matching
  CuPy/CUDA stack.

- NVRTC / header notes: CuPy compiles CUDA kernels at runtime using NVRTC
  which requires the CUDA toolkit headers (for example `cuda_fp16.h`) to
  be available inside the runtime environment. The `-runtime` container
  images provide CUDA drivers and libraries but typically do NOT include
  the CUDA toolkit headers. If you get NVRTC errors like "cannot open
  source file 'cuda_fp16.h'" you have two options:

  - Use a `-devel` CUDA base image (e.g. `nvidia/cuda:12.8.1-devel-ubuntu22.04`)
    in your container so headers and nvcc-compatible tooling are present.

  - Mount or install the CUDA toolkit headers into the container (e.g.
    bind-mount `/usr/local/cuda/include` from the host into the container),
    and ensure `CUDA_HOME`/`CPATH` include that path so NVRTC's compiler
    can find them.

  Additionally, ensure `CUDA_HOME` (commonly `/usr/local/cuda`) is set and
  that `PATH` and `LD_LIBRARY_PATH` include CUDA's `bin` and `lib64`
  respectively so both runtime libraries and headers are discoverable.

Notes
-----

The codebase intentionally removes CPU fallbacks for `src/multiGPU` so
multi-GPU execution requires a functioning CuPy/CUDA runtime. If you need a
CPU-only run for testing, use the `src/baseline` or `src/singleGPU` code
paths instead.

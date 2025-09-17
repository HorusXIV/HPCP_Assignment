Test data workflow
==================

This project excludes the `data/np32/` directory from Git (see `.gitignore`) to
avoid committing large proprietary or binary files. Tests that need a small
representative NPZ stack are supported without pushing data to the repository.

Local development
-----------------
- Use the provided helper to generate a deterministic small test NPZ:

```pwsh
python scripts/generate_test_np32.py --out data/np32/20170906_12_00_12.npz
```

- Alternatively, the test suite will automatically generate a temporary
  synthetic NPZ if `data/np32/20170906_12_00_12.npz` is missing. The
  `HPCP_TEST_INPUT` environment variable may be used to override the input
  path for tests and subprocesses.

CI / Remote runners
-------------------
Choose one of the following approaches in CI jobs, depending on your policy:

1. Generate the synthetic file during the job (recommended):

```yaml
# Example: GitHub Actions step
- name: Generate synthetic test data
  run: |
    python scripts/generate_test_np32.py --out data/np32/20170906_12_00_12.npz
```

2. Download approved dataset from a trusted artifact store (use secure
   credentials):

```yaml
- name: Download real test data
  env:
    ARTIFACT_URL: ${{ secrets.ARTIFACT_URL }}
  run: |
    curl -fSL "$ARTIFACT_URL" -o np32.zip
    unzip np32.zip -d data/np32
```

Notes
-----
- The repository never contains the dataset; CI secrets and artifact stores
  must be managed outside Git.
- Generated data is deterministic by default (seeded) to ensure reproducible
  tests.
- Tests that spawn subprocesses read the `HPCP_TEST_INPUT` env var; the
  session-scoped fixture sets this variable automatically when needed.

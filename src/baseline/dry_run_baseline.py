#!/usr/bin/env python3
"""
Quick test to verify baseline solver works correctly.

Run this to test the baseline module before running full benchmarks.

Usage:
    python test_baseline.py
"""

import numpy as np
from src.baseline import solve_dem, baseline_solver_fn, prepare_inputs


def generate_test_data():
    """Generate minimal test data."""
    print("Generating test data...")

    # Small test case
    H, W = 32, 32
    n_tresp = 50
    n_temps = 51

    # Simulate 6-channel data
    data_6hw = np.random.rand(6, H, W).astype(np.float32) * 1000.0
    data_6hw = np.maximum(data_6hw, 0)

    # Temperature response
    logt_min, logt_max = 5.5, 7.5
    tresp_logt = np.linspace(logt_min, logt_max, n_tresp)

    tresp = np.zeros((n_tresp, 6), dtype=np.float32)
    channel_peaks = [5.8, 6.0, 6.2, 6.4, 6.6, 6.8]
    for i, peak in enumerate(channel_peaks):
        tresp[:, i] = np.exp(-0.5 * ((tresp_logt - peak) / 0.3) ** 2)

    temps = np.linspace(logt_min, logt_max, n_temps)

    return data_6hw, tresp, tresp_logt, temps


def test_solve_dem():
    """Test solve_dem function."""
    print("\n" + "=" * 70)
    print("TEST 1: solve_dem()")
    print("=" * 70)

    data_6hw, tresp, tresp_logt, temps = generate_test_data()

    print(f"Input shapes:")
    print(f"  data_6hw: {data_6hw.shape}")
    print(f"  tresp: {tresp.shape}")
    print(f"  tresp_logt: {tresp_logt.shape}")
    print(f"  temps: {temps.shape}")

    print("\nRunning solve_dem() with validation...")
    try:
        demmap, edemmap, logt, chisq, dn_reg = solve_dem(
            data_6hw,
            tresp,
            tresp_logt,
            temps,
            validate_inputs=True,
            validate_outputs=True,
        )

        print("✓ Solver completed successfully")
        print(f"\nOutput shapes:")
        print(f"  demmap: {demmap.shape}")
        print(f"  edemmap: {edemmap.shape}")
        print(f"  logt: {logt.shape}")
        print(f"  chisq: {chisq.shape}")
        print(f"  dn_reg: {dn_reg.shape}")

        # Check basic properties
        assert demmap.shape == (32, 32, 50), f"Wrong demmap shape: {demmap.shape}"
        assert edemmap.shape == (32, 32, 50), f"Wrong edemmap shape: {edemmap.shape}"
        assert logt.shape == (50,), f"Wrong logt shape: {logt.shape}"
        assert chisq.shape == (32, 32), f"Wrong chisq shape: {chisq.shape}"
        assert dn_reg.shape == (32, 32, 6), f"Wrong dn_reg shape: {dn_reg.shape}"

        print("✓ All output shapes correct")

        # Check for finite values
        finite_frac = np.isfinite(demmap).mean()
        positive_frac = (demmap > 0).mean()
        print(f"\nQuality metrics:")
        print(f"  Finite values: {finite_frac * 100:.1f}%")
        print(f"  Positive values: {positive_frac * 100:.1f}%")
        print(f"  Median chi-square: {np.nanmedian(chisq):.3f}")

        assert finite_frac > 0.9, f"Too many non-finite values: {finite_frac}"

        print("\n✓ TEST 1 PASSED")
        return True

    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_solver_fn():
    """Test baseline_solver_fn (wallclock-compatible)."""
    print("\n" + "=" * 70)
    print("TEST 2: baseline_solver_fn()")
    print("=" * 70)

    data_6hw, tresp, tresp_logt, temps = generate_test_data()

    print("Preparing inputs...")
    data_hw6, edata_hw6 = prepare_inputs(data_6hw)

    print(f"Prepared shapes:")
    print(f"  data_hw6: {data_hw6.shape}")
    print(f"  edata_hw6: {edata_hw6.shape}")

    print("\nRunning baseline_solver_fn()...")
    try:
        demmap, edemmap, logt, chisq, dn_reg = baseline_solver_fn(
            data_hw6,
            edata_hw6,
            tresp,
            tresp_logt,
            temps,
            nmu=42,
        )

        print("✓ Solver completed successfully")
        print(f"\nOutput shapes:")
        print(f"  demmap: {demmap.shape}")
        print(f"  edemmap: {edemmap.shape}")
        print(f"  logt: {logt.shape}")
        print(f"  chisq: {chisq.shape}")
        print(f"  dn_reg: {dn_reg.shape}")

        assert demmap.shape == (32, 32, 50)
        print("✓ Output shape correct")

        print("\n✓ TEST 2 PASSED")
        return True

    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test input validation."""
    print("\n" + "=" * 70)
    print("TEST 3: Input Validation")
    print("=" * 70)

    data_6hw, tresp, tresp_logt, temps = generate_test_data()

    # Test 1: Wrong data shape
    print("\nTest 3.1: Wrong data shape...")
    bad_data = np.random.rand(5, 32, 32)  # Wrong: 5 channels instead of 6
    try:
        solve_dem(bad_data, tresp, tresp_logt, temps, validate_inputs=True)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

    # Test 2: Non-monotonic temps
    print("\nTest 3.2: Non-monotonic temps...")
    bad_temps = temps.copy()
    bad_temps[10] = bad_temps[9] - 0.1  # Make non-monotonic
    try:
        solve_dem(data_6hw, tresp, tresp_logt, bad_temps, validate_inputs=True)
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

    # Test 3: Valid inputs should pass
    print("\nTest 3.3: Valid inputs...")
    try:
        solve_dem(data_6hw, tresp, tresp_logt, temps, validate_inputs=True)
        print("✓ Valid inputs passed validation")
    except Exception as e:
        print(f"✗ Valid inputs failed: {e}")
        return False

    print("\n✓ TEST 3 PASSED")
    return True


def test_error_models():
    """Test different error models."""
    print("\n" + "=" * 70)
    print("TEST 4: Error Models")
    print("=" * 70)

    data_6hw, tresp, tresp_logt, temps = generate_test_data()

    error_models = ["sqrt", "linear", "constant"]

    for model in error_models:
        print(f"\nTesting error_model='{model}'...")
        try:
            demmap, _, _, _, _ = solve_dem(
                data_6hw,
                tresp,
                tresp_logt,
                temps,
                error_model=model,
            )
            print(f"✓ {model} model works")
        except Exception as e:
            print(f"✗ {model} model failed: {e}")
            return False

    print("\n✓ TEST 4 PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BASELINE SOLVER TEST SUITE")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("solve_dem", test_solve_dem()))
    results.append(("baseline_solver_fn", test_baseline_solver_fn()))
    results.append(("validation", test_validation()))
    results.append(("error_models", test_error_models()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(r[1] for r in results)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nBaseline solver is ready to use.")
        print("\nNext steps:")
        print("  1. Run with real data: python -m src.baseline.run --data path/to/data.npz")
        print("  2. Run benchmark: python -m src.baseline.run --benchmark")
        print("  3. Generate goldens: python -m src.baseline.make_goldens")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
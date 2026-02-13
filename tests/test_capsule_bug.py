"""
Reproduce the PyCapsule_GetPointer / SystemError bug.

The bug: torch.as_tensor(__cuda_array_interface__) internally probes for a
versioned DLPack capsule name ("dltensor_versioned"), fails, and leaves a
stale ValueError on the Python error indicator.  CPython then raises:

    SystemError: <function ...> returned a result with an error set

This script tries to trigger it in three ways:
  1. Direct __cuda_array_interface__ wrapping (the exact create_tensor path)
  2. Repeated alloc/free cycles (stress the error-indicator race)
  3. Via vultorch.create_tensor() itself (end-to-end check)

Run:  python tests/test_capsule_bug.py
"""

import sys
import ctypes

def check_stale_error(tag: str):
    """Return True if the Python error indicator is dirty."""
    err = ctypes.py_exc_info_type()
    if err:
        print(f"  [{tag}] STALE ERROR detected on error indicator!")
        ctypes.pythonapi.PyErr_Clear()
        return True
    return False


# ---------- monkey-patch a quick PyErr check via ctypes ----------
ctypes.py_exc_info_type = ctypes.pythonapi.PyErr_Occurred
ctypes.py_exc_info_type.restype = ctypes.py_object if False else ctypes.c_void_p
ctypes.py_exc_info_type.argtypes = []


def test_cuda_array_interface_direct():
    """Wrap a raw CUDA pointer via __cuda_array_interface__ and check for
    stale errors left by torch.as_tensor internals."""
    import torch

    print("\n=== Test 1: Direct __cuda_array_interface__ wrapping ===")
    # Allocate a normal CUDA tensor just to get a valid device pointer
    src = torch.zeros(64, 64, 4, dtype=torch.float32, device="cuda:0")
    ptr = src.data_ptr()

    class _CUDAMem:
        __slots__ = ("__cuda_array_interface__",)
        def __init__(self, p, shape):
            self.__cuda_array_interface__ = {
                "shape": shape,
                "typestr": "<f4",
                "data": (p, False),
                "version": 3,
                "strides": None,
            }

    ctypes.pythonapi.PyErr_Clear()  # start clean

    n_stale = 0
    for i in range(50):
        try:
            mem = _CUDAMem(ptr, (64, 64, 4))
            t = torch.as_tensor(mem, device="cuda:0")
            # Check if error indicator is dirty BEFORE we clear it
            if ctypes.py_exc_info_type():
                n_stale += 1
                if n_stale <= 3:
                    print(f"  iter {i}: stale error on indicator (torch.as_tensor "
                          f"succeeded but left dirty state)")
                ctypes.pythonapi.PyErr_Clear()
            del t
        except Exception as e:
            print(f"  iter {i}: EXCEPTION — {type(e).__name__}: {e}")
            ctypes.pythonapi.PyErr_Clear()
            n_stale += 1

    print(f"  Result: {n_stale}/50 iterations had stale errors or exceptions")
    return n_stale


def test_repeated_alloc_stress():
    """Stress test: rapidly allocate and free shared tensors."""
    import torch

    print("\n=== Test 2: Repeated alloc/free stress ===")
    src = torch.zeros(128, 128, 4, dtype=torch.float32, device="cuda:0")
    ptr = src.data_ptr()

    class _CUDAMem:
        __slots__ = ("__cuda_array_interface__",)
        def __init__(self, p, shape):
            self.__cuda_array_interface__ = {
                "shape": shape,
                "typestr": "<f4",
                "data": (p, False),
                "version": 3,
                "strides": None,
            }

    ctypes.pythonapi.PyErr_Clear()
    n_stale = 0
    tensors = []

    for i in range(200):
        try:
            mem = _CUDAMem(ptr, (128, 128, 4))
            t = torch.as_tensor(mem, device="cuda:0")
            if ctypes.py_exc_info_type():
                n_stale += 1
                ctypes.pythonapi.PyErr_Clear()
            tensors.append(t)
            if len(tensors) > 10:
                tensors.pop(0)
        except Exception as e:
            if n_stale < 3:
                print(f"  iter {i}: {type(e).__name__}: {e}")
            ctypes.pythonapi.PyErr_Clear()
            n_stale += 1

    print(f"  Result: {n_stale}/200 iterations had stale errors or exceptions")
    return n_stale


def test_vultorch_create_tensor():
    """End-to-end test via vultorch.create_tensor()."""
    print("\n=== Test 3: vultorch.create_tensor() end-to-end ===")
    import warnings
    import vultorch

    win = vultorch.Window("capsule-bug-test", 256, 256)

    ctypes.pythonapi.PyErr_Clear()
    n_warnings = 0
    n_errors = 0

    for i in range(20):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t = vultorch.create_tensor(64, 64, channels=4,
                                           device="cuda:0",
                                           name=f"test_{i % 4}",
                                           window=win)
                for warning in w:
                    if "shared GPU memory" in str(warning.message):
                        n_warnings += 1
                        if n_warnings <= 3:
                            print(f"  iter {i}: fallback warning — {warning.message}")

            # Check for stale error after create_tensor
            if ctypes.py_exc_info_type():
                print(f"  iter {i}: stale error AFTER create_tensor (bug not fully fixed!)")
                ctypes.pythonapi.PyErr_Clear()
                n_errors += 1

            del t
        except Exception as e:
            print(f"  iter {i}: EXCEPTION — {type(e).__name__}: {e}")
            ctypes.pythonapi.PyErr_Clear()
            n_errors += 1

    win.destroy()
    print(f"  Result: {n_warnings} fallback warnings, {n_errors} stale errors/exceptions")
    return n_errors


def main():
    print("PyCapsule / DLPack stale-error reproducer")
    print("=" * 55)

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Python:          {sys.version}")

    if not torch.cuda.is_available():
        print("\nNo CUDA — cannot run tests.")
        sys.exit(1)

    total_issues = 0
    total_issues += test_cuda_array_interface_direct()
    total_issues += test_repeated_alloc_stress()

    try:
        total_issues += test_vultorch_create_tensor()
    except Exception as e:
        print(f"\n  Skipped test 3 (vultorch import/window failed): {e}")

    print("\n" + "=" * 55)
    if total_issues == 0:
        print("ALL CLEAN — no stale errors detected on this platform.")
    else:
        print(f"TOTAL ISSUES: {total_issues}")
        print("The stale-error bug is reproducible on this platform.")
        print("With the try/except fallback, create_tensor() should still work")
        print("(check test 3 output for fallback warnings).")

    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()

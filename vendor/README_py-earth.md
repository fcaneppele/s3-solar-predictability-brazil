# Vendored py-earth (MARS)

Source: https://github.com/scikit-learn-contrib/py-earth (BSD-3-Clause, J. Rudy), version 0.1.0.
Patched for current Python/NumPy/scikit-learn, without changing the algorithm:
- `np.float`, `np.int`, `np.bool` aliases replaced by `np.float64`, `np.int64`, `np.bool_`;
- Cython types `cnp.int_t` replaced by `cnp.int64_t` (removed in NumPy 2);
- Cython profiling directive disabled (incompatible with Python >= 3.11);
- `versioneer.py`: `SafeConfigParser`/`readfp` replaced by `ConfigParser`/`read_file`;
- `earth.py`: shim for the scikit-learn argument rename `force_all_finite` -> `ensure_all_finite`.

Install (requires a C compiler, Cython < 3 and setuptools < 70):
    pip install "cython<3" "setuptools<70" wheel
    cd vendor/py-earth && python setup.py build_ext --inplace --cythonize
    # then add vendor/py-earth to PYTHONPATH (the notebook does it automatically)
Tested on Python 3.10 (NumPy 1.23, scikit-learn 1.2.2) and Python 3.12 (NumPy 2.2, scikit-learn 1.9).
- `setup.py`: version fixed to 0.1.0.post1 (versioneer needs git metadata, absent in a vendored copy).
- `np.linalg.lstsq` called with `rcond=-1` (the NumPy < 2 default, so results are identical across NumPy versions); empty-array truth tests replaced by explicit size checks.

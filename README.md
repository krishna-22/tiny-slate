# SLATE

**Sparse Lightweight Additive Threshold Ensemble**, a tiny, fully interpretable multinomial classifier.


## Files

| File | What it does |
|---|---|
| `slate_shared.py` | The model. The `SlateShared` classifier (`fit`, `predict`, `predict_proba`) plus its footprint accounting. |
| `slate_benchmark.py` | Reproduces the accuracy, size, and latency comparison against XGBoost, EBM, FIGS, RuleFit, random forests, and other baselines across 16 OpenML datasets. |
| `slate_ondevice.py` | On-device evaluation. Exports SLATE to C, cross-compiles for ARM Cortex M0 and M4F, and measures flash, RAM, instruction counts, and latency under QEMU. |

## Usage

```python
from slate_shared import SlateShared

clf = SlateShared(budget=64).fit(X_train, y_train)
clf.predict(X_test)
print(clf.n_atoms_, clf.memory_bytes_)   # model size
```

Run the benchmark (downloads datasets from OpenML; a tiny validation run is available via the SMOKE environment variable):

```bash
python slate_benchmark.py             # full run
SMOKE=1 python slate_benchmark.py     # quick smoke run
```

The on-device evaluation needs the ARM toolchain and QEMU:

```bash
python slate_ondevice.py bootstrap    # install arm gcc, qemu, emlearn
python slate_ondevice.py all
```

## Requirements

- Python 3.9 or newer, numpy, scikit-learn
- Benchmark: pandas, xgboost, interpret, imodels (auto-installed on first run)
- On-device: the ARM bare-metal gcc toolchain, qemu-system-arm, emlearn

A packaged, scikit-learn compatible version of the model is available on PyPI as `tinyslate` (`pip install tinyslate`).

## License

MIT

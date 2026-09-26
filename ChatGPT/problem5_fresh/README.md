# Problem 5: fresh image classifier

Generated from the specified Géron Chapter 10 image-classifier section during this session. No prior classifier implementation was copied or imported. `INITIAL_PROMPT.md` records the single initiating prompt; the dialogue summary documents the actual process and results.

## Reproduce

Use Python 3.12 (tested) and the pinned packages in `requirements.txt`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
# Optional: reuse a directory containing FashionMNIST/raw:
# export FASHION_MNIST_ROOT=/absolute/path/to/datasets
python build_notebook.py
python execute_notebook.py
```

The default dataset location is `datasets/`; TorchVision downloads data if needed. A Jupyter kernel requires permission to open local sockets. Run from a writable folder. The execution helper starts a clean kernel, executes all ten code cells, validates outputs and exact script equality, and exports `SO_GPT_Problem5_Image_Classifier.html`. Opening `SO_GPT_Problem5_Image_Classifier.ipynb` and selecting Restart Kernel / Run All is also supported. The notebook contains all stage code and does not import the stage files. The scripts are sequential stages sharing one namespace, not standalone programs.

## Configuration and interpretation

Fashion-MNIST uses 55,000 training, 5,000 validation, and 10,000 held-out test examples; float32 pixels scaled to [0,1]; batch size 32; seed 42; Flatten → Linear(784,300) → ReLU → Linear(300,100) → ReLU → Linear(100,10); cross-entropy; SGD at 0.1 with default zero momentum; 20 complete epochs. The reference's `n_epochs` comes from earlier cell 63. Helpers in cells 85, 91, and 93 were inspected.

Justified deviations: deterministic CPU execution with two threads for portable reproducibility; accuracy computed as correct/total rather than with TorchMetrics (same multiclass micro accuracy); evaluation loss weighted by sample count. Training loss retains the reference's average of batch losses. Training accuracy records predictions during parameter updates, while validation accuracy uses the final model each epoch. Additional plots and final held-out test evaluation satisfy this assignment. The first three validation predictions and top-four conditional softmax follow the reference. Conditional top-four probabilities are distinct from full ten-class probabilities.

Exact results, package versions, duration, and checks are in `artifacts/results.json`, `history.json`, and `verification.json`. Two PNG figures are also embedded in the notebook and HTML. This is one seed and one CPU run; numerical results may differ across package versions or hardware. No tuning or checkpoint selection used the test set. Model weights are not needed to reproduce the notebook and are not included.

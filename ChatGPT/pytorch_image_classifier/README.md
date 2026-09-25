# Fashion-MNIST classifier — OpenAI Codex assignment

This self-contained adaptation follows the “Building an Image Classifier with PyTorch” section of the supplied Chapter 10 notebook by Aurélien Géron (2025). It adds the requested training-accuracy plot, validation curves, held-out test evaluation, and model save/reload verification.

## Submission files

- `image_classifier.ipynb`: notebook with seven ordered, labeled code cells and recorded outputs.
- `image_classifier.html`: rendered notebook with embedded plots (the requested HTML image/export).
- `dialogue_summary.md`: factual summary of this conversation and generated work.
- `scripts/01_...py` through `scripts/07_...py`: exact code used in the seven cells.
- `artifacts/`: learning curves, sample/prediction images, per-epoch CSV metrics, final JSON results, and saved model weights.

## Reproduce

Use Python 3.12 for the pinned dependencies. Create and activate a virtual environment, then run from this directory:

```bash
python -m pip install -r requirements.txt
python build_notebook.py
python execute_notebook.py
```

The first run downloads Fashion-MNIST and requires internet access. The execution utility uses the same Python environment as the command and creates a local, ignored kernel specification. It runs every notebook cell and regenerates the HTML. The HTML and executed notebook can be viewed without rerunning training. `requirements-lock.txt` records the complete submitted environment; cross-platform dependency availability can differ.

For execution without a notebook:

```bash
MPLBACKEND=Agg python run_all.py
```

The numbered stage scripts share a namespace and must run in order; they are not independent command-line programs. The notebook embeds their source verbatim, so a downloaded notebook remains usable without separate script imports. If you edit a script, rebuild and execute the notebook to refresh the code and results.

## Evaluation design

Train/validation/test sizes are 55,000/5,000/10,000. Hyperparameters follow the reference: batch size 32, SGD learning rate 0.1, and 20 epochs. The 784–300–100–10 MLP has 266,610 parameters. The checkpoint with highest validation accuracy is evaluated on the held-out test set. Training accuracy is accumulated during each epoch; validation accuracy is computed after it. Losses are sample-weighted. CPU, fixed seeds, and deterministic operations support reproducibility, but exact results depend on the software and hardware environment.

This submission demonstrates the OpenAI/Codex portion of the assignment. It is not a Claude execution or a controlled comparison of the two systems. No synthetic dialogue or invented execution results are included.

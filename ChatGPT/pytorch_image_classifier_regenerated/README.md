# Regenerated PyTorch image classifier

Freshly assembled and executed from the previous assignment scripts on September 25, 2026. The seven labeled code cells train the Fashion-MNIST classifier for 20 epochs and include training/validation accuracy and loss plots, test predictions, and checkpoint reload verification.

## Submission files
- `image_classifier.ipynb`: executed notebook (identical to the delivered `image_classifier_new.ipynb`).
- `image_classifier.html`: HTML rendering with three embedded plots (identical to the delivered `image_classifier_new.html`).

## Reproduction
Install `requirements.txt` using Python 3.12. Run `python build_notebook.py`, then `python execute_notebook.py`, or open the notebook and run all cells in order. The first run downloads Fashion-MNIST. `execute_inprocess.py` offers an IPython execution alternative without a separate kernel server.

## Verified results
All seven code cells completed without errors. Final training accuracy: 92.86%; best validation accuracy: 89.10% at epoch 19; test accuracy: 88.73%. The notebook and HTML were freshly executed using IPython in process because the sandbox blocked Jupyter kernel socket creation. Model weights, metrics, and plots are included in `artifacts/`; downloaded datasets and environment caches are excluded.

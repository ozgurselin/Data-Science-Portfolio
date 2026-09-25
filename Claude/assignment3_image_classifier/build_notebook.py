# Script 10 - Assemble scripts/01-09 into an executable, labeled Jupyter notebook
import glob
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))

DESCRIPTIONS = {
    "01_setup.py": "Imports, version checks, plotting defaults and device selection.",
    "02_load_dataset.py": "Download Fashion MNIST with TorchVision; split 55,000 train / 5,000 validation.",
    "03_dataloaders.py": "Wrap the datasets in DataLoaders and inspect one sample.",
    "04_model.py": "Define the two-hidden-layer MLP `ImageClassifier` and the cross-entropy loss.",
    "05_train_functions.py": "The book's `evaluate_tm()` and `train2()` training loop.",
    "06_train.py": "Train for 20 epochs with SGD (lr=0.1), keeping the returned `history`.",
    "07_plot_training_accuracy.py": "**New code (not in the book):** plot the training accuracy per epoch.",
    "08_predictions.py": "Predict classes and probabilities for three validation images.",
    "09_test_accuracy.py": "Evaluate the trained model on the test set.",
}

nb = new_notebook()
nb.cells.append(new_markdown_cell(
    "# Assignment 3 – Building an Image Classifier with PyTorch\n\n"
    "Rebuilds the *Building an Image Classifier with PyTorch* section of "
    "`10_neural_nets_with_pytorch.ipynb` (Géron, *Hands-On Machine Learning with "
    "Scikit-Learn and PyTorch*, 2025) using scripts generated incrementally by "
    "Claude Code, and adds a plot of the training accuracy.\n\n"
    "Each code cell below is one generated script from "
    "`Claude/assignment3_image_classifier/scripts/`; the markdown cell above it "
    "names the script."))

for path in sorted(glob.glob(os.path.join(HERE, "scripts", "0*.py"))):
    name = os.path.basename(path)
    number = name.split("_")[0]
    nb.cells.append(new_markdown_cell(
        f"### Cell {int(number)} — generated script `scripts/{name}`\n\n"
        f"{DESCRIPTIONS[name]}"))
    with open(path) as f:
        nb.cells.append(new_code_cell(f.read().rstrip()))

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
out = os.path.join(HERE, "Assignment3_Image_Classifier.ipynb")
nbformat.write(nb, out)
print("wrote", out)

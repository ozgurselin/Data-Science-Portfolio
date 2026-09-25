"""Assemble scripts/01-09 into a labeled, self-contained Jupyter notebook."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

INTRO = """# Fashion MNIST Image Classifier with PyTorch

**Assignment 3 - Problem 3.** Problems 1 and 2 redone by Claude Code from one long prompt in a new
session. The notebook rebuilds the *Building an Image Classifier with PyTorch* section of Chapter 10 of
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron, 2025) and adds a plot of the
training accuracy.

Each code cell is one generated script from `scripts/`, run in order. The notebook is self-contained:
the imports, device selection and the `evaluate_tm()` / `train2()` helpers from earlier in the
chapter are included, so it runs from top to bottom on its own.

| Cell | Script | Step |
|---|---|---|
""" 

STEPS = [
    ("01_setup.py", "Setup",
     "Import the libraries, check versions, set plot defaults and pick the fastest device "
     "(CUDA GPU, Apple-silicon MPS, or CPU)."),
    ("02_load_dataset.py", "Loading the dataset with TorchVision",
     "Fashion MNIST has 70,000 grayscale 28×28 images in 10 classes. Each image becomes a "
     "`float32` tensor in [0, 1]; the 60,000 training images are split 55,000 / 5,000 into "
     "training and validation sets."),
    ("03_dataloaders.py", "Data loaders and sample images",
     "Batches of 32 images. Each entry is an `(image, target)` tuple and each image has shape "
     "`[channels, rows, columns]` = `[1, 28, 28]`."),
    ("04_model.py", "Building the classifier",
     "An MLP: flatten to 784 inputs, two ReLU hidden layers (300 and 100 units), and 10 output "
     "logits. `nn.CrossEntropyLoss` applies the softmax itself."),
    ("05_train_functions.py", "Training and evaluation helpers",
     "`evaluate_tm()` computes a `torchmetrics` metric over a data loader; `train2()` runs the "
     "training loop and returns the per-epoch loss and metrics. Both come from earlier in Chapter 10."),
    ("06_train.py", "Training the model",
     "20 epochs of SGD (learning rate 0.1), tracking multiclass accuracy. The book discards "
     "`train2()`'s return value; here it is kept as `history` for plotting."),
    ("07_plot_training_accuracy.py", "Plotting the training accuracy (new)",
     "Not in the book. Training accuracy is averaged while the weights change during an epoch, so "
     "it is plotted half an epoch earlier. Validation accuracy and training loss are shown for "
     "comparison."),
    ("08_predictions.py", "Making predictions",
     "Classify three validation images: the predicted class is the largest logit, and the softmax "
     "of the logits gives the class probabilities."),
    ("09_test_accuracy.py", "Evaluating on the test set",
     "How well the model generalizes to the 10,000 held-out test images."),
]


def main():
    intro = INTRO + "\n".join(f"| {i} | `{name}` | {title} |"
                              for i, (name, title, _) in enumerate(STEPS, 1))
    cells = [nbf.v4.new_markdown_cell(intro)]
    for i, (name, title, text) in enumerate(STEPS, 1):
        cells.append(nbf.v4.new_markdown_cell(
            f"## Cell {i} - {title}\n*Generated script `scripts/{name}`*\n\n{text}"))
        cells.append(nbf.v4.new_code_cell((HERE / "scripts" / name).read_text().strip()))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    out = HERE / "Problem3_Fashion_MNIST_Classifier.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()

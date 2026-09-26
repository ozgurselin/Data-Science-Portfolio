from pathlib import Path
import nbformat
root = Path(__file__).resolve().parent
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [nbformat.v4.new_markdown_cell("# Problem 5 — Fresh PyTorch image classifier\nReference: Géron (2025), Chapter 10, ‘Building an Image Classifier with PyTorch’, cells 111–132; helpers 85, 91, 93 and 20 epochs from cell 63.\n\nPreserved: scaled float32 images, seeded 55,000/5,000 split, batch size 32, 784→300→100→10 ReLU MLP, cross-entropy, SGD learning rate 0.1, 20 epochs. Deviations: deterministic CPU with two threads for portable execution; direct correct/total accuracy replaces TorchMetrics equivalently; evaluation loss is sample-weighted. Training loss retains the reference batch mean, and training accuracy measures predictions during updates. Added held-out test evaluation and plots. No earlier classifier implementation was used.")]
for script in sorted(root.glob("[0-9][0-9]_*.py")):
    nb.cells += [nbformat.v4.new_markdown_cell("## " + script.name), nbformat.v4.new_code_cell(script.read_text())]
nbformat.write(nb, root / "SO_GPT_Problem5_Image_Classifier.ipynb")

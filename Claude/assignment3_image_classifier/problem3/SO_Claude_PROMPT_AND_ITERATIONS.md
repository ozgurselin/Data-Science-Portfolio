# Problem 3 - Single-prompt run

## The prompt (new session)
> Problem 3. Start a new session. This time ask Claude/code in a longer single prompt to accomplish
> all it did in Problem 1 and 2. It is alright to ask for corrections (iterate). Submit newly
> generated notebook and its HTML image.

That covers everything from Problems 1 and 2: rebuild the *Building an Image Classifier with PyTorch*
section of Géron ch. 10 (`Claude/10_neural_nets_with_pytorch.ipynb`, cells 111-132) step by step as
separate scripts, add a training-accuracy plot, include the helpers the section needs from earlier in
the chapter (`device`, `n_epochs`, `evaluate_tm`, `train2`), combine the scripts into one labeled,
self-contained notebook, run it end to end, export to HTML, and commit to GitHub.

## What Claude did
1. Read the chapter notebook section and its dependencies (cells 37, 63, 91, 93).
2. Wrote `scripts/01-09`, one per step (setup → data → loaders → model → helpers → train →
   **new training-accuracy plot** → predictions → test accuracy).
3. Wrote `build_notebook.py`, which puts each script into a code cell under a markdown heading
   naming the script and explaining the step, plus an index table at the top.
4. Ran the notebook with `jupyter nbconvert --execute` (~2 min on Apple-silicon MPS) and exported HTML.

## Iterations (corrections)
1. First run worked, but the output had a torchvision `libjpeg` warning and dataset download
   progress bars. Added a `warnings.filterwarnings` call; the dataset is now cached, so the bars are gone.
2. The warning still showed because `torchmetrics` imports torchvision first, so the filter was
   moved above all the imports. After re-running, the HTML output is clean.

## Results (20 epochs, seed 42)
- Training accuracy 78.1% → **92.8%**; validation best 88.9% (epoch 12), final 88.0%
- **Test accuracy 88.3%**; 3/3 sample validation images classified correctly; 266,610 parameters

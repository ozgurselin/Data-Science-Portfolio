from pathlib import Path
import json, hashlib, time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
root = Path(__file__).resolve().parent
path = root / "SO_GPT_Problem5_Image_Classifier.ipynb"
nb = nbformat.read(path, as_version=4)
start = time.perf_counter()
ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(root)}})
nbformat.write(nb, path)
code = [c for c in nb.cells if c.cell_type == "code"]
scripts = sorted(root.glob("[0-9][0-9]_*.py"))
assert [c.execution_count for c in code] == list(range(1, len(code)+1))
assert all(c.source == p.read_text() for c,p in zip(code,scripts)) and len(code)==len(scripts)
assert not any(o.output_type == "error" for c in code for o in c.outputs)
plots = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
assert plots >= 2
html, _ = HTMLExporter().from_notebook_node(nb)
assert "data:image/png;base64," in html
path.with_suffix(".html").write_text(html)
report = {"method": "nbconvert ExecutePreprocessor; fresh Python kernel; all cells executed", "execution_counts": [c.execution_count for c in code], "error_outputs": 0, "embedded_png_outputs": plots, "script_cells_exact_match": True, "execution_and_export_seconds": time.perf_counter()-start, "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}}
(root / "artifacts" / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

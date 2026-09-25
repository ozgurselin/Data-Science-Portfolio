"""Execute with the current Python environment and export self-contained HTML."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT = Path(__file__).resolve().parent
# Use a project-local kernel specification, without changing the user's kernels.
kernel_root = ROOT / ".jupyter"
kernel_dir = kernel_root / "kernels" / "assignment"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Assignment Python", "language": "python"}))
os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter" / "ipython"))
nb_path = ROOT / "image_classifier.ipynb"
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="assignment", resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(nb, nb_path)
html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
(ROOT / "image_classifier.html").write_text(html)
print("Executed notebook and exported image_classifier.html")

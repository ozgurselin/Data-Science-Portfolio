"""Execute the seven generated stages in their shared namespace."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
namespace = {"__name__": "__main__"}
for path in sorted((ROOT / "scripts").glob("[0-9][0-9]_*.py")):
    print(f"\nRunning {path.name}", flush=True)
    exec(compile(path.read_text(), str(path), "exec"), namespace)

from pathlib import Path
import os
root=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(root/'.mplconfig')
os.environ['IPYTHONDIR']=str(root/'.ipython')
os.chdir(root)
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
from nbconvert import HTMLExporter
shell=TerminalInteractiveShell.instance()
shell.run_cell('%matplotlib inline')
nb=nbformat.read('image_classifier.ipynb',as_version=4)
i=0
for cell in nb.cells:
 if cell.cell_type!='code': continue
 i+=1
 with capture_output() as captured:
  result=shell.run_cell(cell.source,store_history=True)
 cell.execution_count=i
 cell.outputs=[]
 if captured.stdout: cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
 if captured.stderr: cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
 for output in captured.outputs:
  cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
 if result.error_before_exec or result.error_in_exec: raise RuntimeError(f'Cell {i} failed')
 print(f'Cell {i} completed',flush=True)
nbformat.validate(nb)
nbformat.write(nb,'image_classifier.ipynb')
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
Path('image_classifier.html').write_text(html)
print('Export complete',flush=True)

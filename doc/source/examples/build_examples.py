import os
from glob import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Execute all notebooks in the examples directory manually.
# Saves outputs in place
# References:
# https://nbconvert.readthedocs.io/en/latest/execute_api.html
# https://nbsphinx.readthedocs.io/en/0.5.1/pre-executed.html

# Get all the notebooks in the examples directory
example_dir = os.path.dirname(os.path.abspath(__file__))
examples = glob(os.path.join(example_dir, "*.ipynb"))

for example in examples:
    print("Executing: ", os.path.basename(example))

    # Read the notebook
    with open(example) as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook (30 min timeout)
    ep = ExecutePreprocessor(timeout=1800)
    ep.preprocess(nb, {'metadata': {'path': example_dir}})

    # Save the executed notebook
    with open(example, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

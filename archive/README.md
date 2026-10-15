# Archive: Legacy Notebook Patch Scripts

This directory contains historical patch and inspection scripts that were used during earlier development of `Student_Grade_Prediction.ipynb`:

1. `fix_notebook.py`: Removed Google Colab drive mount in Cell 1 and preserved G1/G2 in Cell 24.
2. `patch_notebook.py`: Converted categorical variables with `pd.get_dummies(..., dtype=int)` in Cell 20, and added `numeric_only=True` to correlation calls in Cells 23 and 25.
3. `update_notebook.py`: Diagnostic utility for listing notebook cells.

### Audit Confirmation:
All modifications made by these scripts have already been permanently applied and verified directly within `Student_Grade_Prediction.ipynb`. These scripts are retained here strictly for historical auditability and are not required to train, test, or run GradeSense.

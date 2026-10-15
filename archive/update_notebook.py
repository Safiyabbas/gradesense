import json

notebook_path = 'Student_Grade_Prediction.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get("source", []))
    print(f"{i}: {cell.get('cell_type')} - {repr(source[:60])}")

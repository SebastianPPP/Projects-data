import csv
from itertools import islice
from pathlib import Path


data_path = Path(__file__).resolve().parent / "data.csv"

with data_path.open(newline="", encoding="utf-8-sig") as csv_file:
    reader = csv.DictReader(csv_file)
    columns = reader.fieldnames or []
    preview = list(islice(reader, 5))
    row_count = len(preview) + sum(1 for _ in reader)

print(f"Loaded {data_path.name}: {row_count} rows, {len(columns)} columns")
print("Columns:", ", ".join(columns))
print("First 5 rows:")
for row in preview:
    print(row)
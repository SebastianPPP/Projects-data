import csv
from itertools import islice
from pathlib import Path


data_directory = Path(__file__).resolve().parent


def show_csv_preview(filename):
    csv_path = data_directory / filename

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        preview = list(islice(reader, 5))
        row_count = len(preview) + sum(1 for _ in reader)

    print(f"\nLoaded {csv_path.name}: {row_count} rows, {len(columns)} columns")
    print("Columns:", ", ".join(columns))
    print("First 5 rows:")
    for row in preview:
        print(row)


show_csv_preview("satellite_image.csv")
show_csv_preview("labels.csv")
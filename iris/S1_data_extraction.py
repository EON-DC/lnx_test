import os
from pathlib import Path
import __main__
import pandas as pd
import numpy as np

DATABASE_PATH = Path(__main__.__file__).resolve().parent / "input/database.sqlite"
CSV_PATH = Path(__main__.__file__).resolve().parent / "input/Iris.csv"
OUTPUT_DIR = Path(__main__.__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_data():
    import sqlite3

    conn = sqlite3.connect(DATABASE_PATH)

    # Extracting data from the database
    df = pd.read_sql_query("SELECT * FROM iris", conn)

    print(f"Data extracted from {DATABASE_PATH}")
    print(df.shape)

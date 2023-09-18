# data_ingestion.py

import sqlite3
import pandas as pd

def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return data

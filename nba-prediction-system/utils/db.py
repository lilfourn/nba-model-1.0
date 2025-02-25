
import sqlite3
import pandas as pd
import logging
from contextlib import contextmanager
from config.paths import *

@contextmanager
def db_connection(db_path):
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def query_to_df(db_path, query, params=None):
    """Execute query and return results as DataFrame"""
    with db_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)

def save_df_to_db(df, db_path, table_name, if_exists='replace'):
    """Save DataFrame to SQLite database"""
    with db_connection(db_path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        logging.info(f"Saved {len(df)} records to {table_name} in {db_path}")

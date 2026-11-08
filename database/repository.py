import pandas as pd

from database.connection import get_connection


def upsert_dataframe(df: pd.DataFrame, table: str, pk_cols: list[str]) -> None:
    """Insert-or-replace rows of df into table, keyed on pk_cols."""
    if df.empty:
        return
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    conn = get_connection()
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    conn.executemany(sql, df[cols].itertuples(index=False, name=None))
    conn.commit()
    conn.close()


def read_table(table: str, where: str = "", params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

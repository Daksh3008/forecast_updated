# db/init_db.py

from db.connection import get_conn
from db.schema import CREATE_TABLE_SQL

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.close()
    conn.close()
    print("[OK] Database initialized")

if __name__ == "__main__":
    init_db()

import os

try:
    import mysql.connector
    from mysql.connector import pooling
except ImportError:  # lets the API run (without history) if the package is missing
    mysql = None

_pool = None


def _get_pool():
    global _pool
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed")
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="seg", pool_size=5,
            host=os.getenv("DB_HOST", "localhost"), port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"), password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "tumor_seg"),
        )
    return _pool


def _run(sql, args=(), fetch=None, commit=False):
    conn = _get_pool().get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, args)
        out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else cur.lastrowid
        if commit:
            conn.commit()
        cur.close()
        return out
    finally:
        conn.close()


def _clean(row):
    if row and row.get("created_at"):
        row["created_at"] = row["created_at"].isoformat()
    return row


def insert_prediction(d):
    return _run(
        """INSERT INTO predictions (original_name,image_file,mask_file,overlay_file,width,height,
           tumor_pixels,tumor_percent,threshold_val,demo_mode,inference_ms)
           VALUES (%(original_name)s,%(image_file)s,%(mask_file)s,%(overlay_file)s,%(width)s,%(height)s,
           %(tumor_pixels)s,%(tumor_percent)s,%(threshold_val)s,%(demo_mode)s,%(inference_ms)s)""",
        d, commit=True)


def list_predictions(limit=50):
    return [_clean(r) for r in _run("SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,), fetch="all")]


def get_prediction(pid):
    return _clean(_run("SELECT * FROM predictions WHERE id=%s", (pid,), fetch="one"))


def delete_prediction(pid):
    _run("DELETE FROM predictions WHERE id=%s", (pid,), commit=True)

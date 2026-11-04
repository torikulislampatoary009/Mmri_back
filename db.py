import os
import tempfile
from urllib.parse import unquote, urlparse

try:
    import mysql.connector
    from mysql.connector import pooling
except ImportError:  # API still runs (without history) if the package is missing
    mysql = None

_pool = None

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS predictions (
         id            INT AUTO_INCREMENT PRIMARY KEY,
         original_name VARCHAR(255) NOT NULL,
         width         INT NOT NULL,
         height        INT NOT NULL,
         tumor_pixels  INT NOT NULL,
         tumor_percent FLOAT NOT NULL,
         threshold_val FLOAT NOT NULL,
         demo_mode     TINYINT(1) NOT NULL DEFAULT 0,
         inference_ms  INT NOT NULL,
         created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
         INDEX idx_created (created_at)
       )""",
    # Images live in the database because Render's free disk is wiped on every restart.
    """CREATE TABLE IF NOT EXISTS prediction_files (
         prediction_id INT NOT NULL,
         kind          VARCHAR(10) NOT NULL,
         data          MEDIUMBLOB NOT NULL,
         PRIMARY KEY (prediction_id, kind),
         FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
       )""",
]


def _config():
    """Connection settings from DB_URL (Aiven 'Service URI') or from separate DB_* variables."""
    url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if url:
        u = urlparse(url)
        cfg = dict(host=u.hostname, port=u.port or 3306, user=unquote(u.username or ""),
                   password=unquote(u.password or ""), database=(u.path or "").lstrip("/") or None)
    else:
        cfg = dict(host=os.getenv("DB_HOST", "localhost"), port=int(os.getenv("DB_PORT", 3306)),
                   user=os.getenv("DB_USER", "root"), password=os.getenv("DB_PASSWORD", ""),
                   database=os.getenv("DB_NAME", "tumor_seg"))

    ca = os.getenv("DB_SSL_CA")
    pem = os.getenv("DB_SSL_CA_PEM")
    if pem:  # certificate pasted into an environment variable
        ca = os.path.join(tempfile.gettempdir(), "db-ca.pem")
        with open(ca, "w") as f:
            f.write(pem.replace("\\n", "\n"))
    if os.getenv("DB_SSL", "true").lower() == "false":
        cfg["ssl_disabled"] = True
    else:  # Aiven requires TLS
        cfg["ssl_disabled"] = False
        if ca:
            cfg["ssl_ca"] = ca
            cfg["ssl_verify_cert"] = True
    cfg["connection_timeout"] = 15
    return cfg


def _get_pool():
    global _pool
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed")
    if _pool is None:
        cfg = _config()
        conn = mysql.connector.connect(**cfg)  # fail fast with a clear error, and create tables once
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()
        _pool = pooling.MySQLConnectionPool(pool_name="seg", pool_size=3, **cfg)
    return _pool


def _conn():
    conn = _get_pool().get_connection()
    conn.ping(reconnect=True, attempts=3, delay=1)  # idle connections go stale on free tiers
    return conn


def _clean(row):
    if row and row.get("created_at"):
        row["created_at"] = row["created_at"].isoformat()
    return row


def ping():
    conn = _conn()
    conn.close()


def insert_prediction(meta, files):
    """meta: dict of column values. files: {kind: bytes}. Saved in one transaction."""
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO predictions (original_name,width,height,tumor_pixels,tumor_percent,
               threshold_val,demo_mode,inference_ms)
               VALUES (%(original_name)s,%(width)s,%(height)s,%(tumor_pixels)s,%(tumor_percent)s,
               %(threshold_val)s,%(demo_mode)s,%(inference_ms)s)""", meta)
        pid = cur.lastrowid
        for kind, data in files.items():
            cur.execute("INSERT INTO prediction_files (prediction_id,kind,data) VALUES (%s,%s,%s)", (pid, kind, data))
        conn.commit()
        cur.close()
        return pid
    finally:
        conn.close()


def _fetch(sql, args=(), one=False):
    conn = _conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, args)
        out = cur.fetchone() if one else cur.fetchall()
        cur.close()
        return out
    finally:
        conn.close()


def list_predictions(limit=50):
    return [_clean(r) for r in _fetch("SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,))]


def get_prediction(pid):
    return _clean(_fetch("SELECT * FROM predictions WHERE id=%s", (pid,), one=True))


def get_file(pid, kind):
    row = _fetch("SELECT data FROM prediction_files WHERE prediction_id=%s AND kind=%s", (pid, kind), one=True)
    return bytes(row["data"]) if row else None


def delete_prediction(pid):
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions WHERE id=%s", (pid,))  # files go with it (ON DELETE CASCADE)
        conn.commit()
        cur.close()
    finally:
        conn.close()

"""The fake workspace: one SQLite DB per episode, built from a seed, dumpable at the end."""
import sqlite3
from datetime import datetime

# The whole world lives in these tables. Constraints are deliberate: the DB itself rejects
# nonsense (unknown owner, priority "P0") so neither tools nor verifiers have to re-check it.
SCHEMA = """
CREATE TABLE meta     (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE users    (id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE accounts (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                       owner_id TEXT NOT NULL REFERENCES users(id));
CREATE TABLE messages (id INTEGER PRIMARY KEY, channel TEXT NOT NULL,
                       account_id TEXT REFERENCES accounts(id),
                       author TEXT NOT NULL, text TEXT NOT NULL, ts TEXT NOT NULL);
CREATE TABLE tickets  (id INTEGER PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
                       title TEXT NOT NULL,
                       priority TEXT NOT NULL CHECK (priority IN ('P1','P2','P3')),
                       -- NULL = unassigned (valid, D23); a non-NULL owner must still exist.
                       owner_id TEXT REFERENCES users(id), created_at TEXT NOT NULL);
"""
# Insert order matters: parents before children, or foreign keys fail.
TABLES = ["users", "accounts", "messages", "tickets"]


def _aware(ts: str, what: str) -> datetime:
    """Parse an ISO timestamp and require a timezone. Naive vs aware comparisons raise
    TypeError, which callers filtering bad seeds with `except ValueError` would not catch."""
    dt = datetime.fromisoformat(ts)  # bad format already raises ValueError
    if dt.tzinfo is None:
        raise ValueError(f"{what}: timestamp {ts!r} has no timezone")
    return dt


def load_seed(seed: dict) -> sqlite3.Connection:
    """Fresh in-memory DB holding exactly the seed. Called on every reset()."""
    # Validate the seed dict before touching SQLite: fail fast, all checks in one place.
    seed_now = _aware(seed["now"], "now")
    for message in seed.get("messages", []):
        # A message from the future makes the task incoherent ("this week" relative to what?).
        if _aware(message["ts"], f"message {message.get('id')}") > seed_now:
            raise ValueError(f"message {message.get('id')}: ts is later than now")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # SQLite ignores REFERENCES unless this is on, per connection. Easy silent bug.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # The fixed clock lives in the DB, so tools only need `conn` and the dump carries it too.
    conn.execute("INSERT INTO meta VALUES ('now', ?)", (seed["now"],))
    for table in TABLES:
        allowed = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for row in seed.get(table, []):
            # Seeds will come from an LLM (Phase 4). Column names go into the SQL string,
            # so whitelist them; values always go through ? placeholders.
            if not set(row) <= allowed:
                raise ValueError(f"{table}: unknown columns {set(row) - allowed}")
            cols, marks = ", ".join(row), ", ".join("?" * len(row))
            conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
    conn.commit()
    return conn


def now(conn: sqlite3.Connection) -> datetime:
    """The episode's clock. Tools call this instead of datetime.now()."""
    return datetime.fromisoformat(conn.execute("SELECT value FROM meta WHERE key='now'").fetchone()[0])


def dump_state(conn: sqlite3.Connection) -> dict:
    """Whole DB as plain JSON-able dicts. Stored in every trajectory so verifiers can
    re-score episodes offline without replaying them."""
    state = {"now": now(conn).isoformat()}
    for table in TABLES:
        # ORDER BY rowid: same DB -> byte-identical dump, so dumps can be diffed/hashed.
        state[table] = [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
    return state

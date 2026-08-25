from fastmcp import FastMCP
from datetime import date as _date
import os
import aiosqlite
import sqlite3
import tempfile

# Temp dir is writable on the hosted container; override with EXPENSE_DB_PATH
# to point at durable storage (a mounted volume, Turso, etc.).
DB_PATH = os.environ.get(
    "EXPENSE_DB_PATH",
    os.path.join(tempfile.gettempdir(), "expenses.db"),
)
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

print(f"Database path: {DB_PATH}")

mcp = FastMCP("Expense_Tracker")

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
        """)

        # Test write access
        c.execute("INSERT OR IGNORE INTO expenses(date, amount, category) VALUES ('2000-01-01', 0, 'test')")
        c.execute("DELETE FROM expenses WHERE category = 'test'")
        print("Database initialized successfully with write access")

init_db()


def _check_date(value, field):
    '''Return value as YYYY-MM-DD, or raise if it is not a real date.'''
    try:
        return _date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")


def _check_range(start_date, end_date):
    '''Validate both ends of a date range and reject a reversed range.'''
    start_date = _check_date(start_date, "start_date")
    end_date = _check_date(end_date, "end_date")
    if start_date > end_date:
        raise ValueError(
            f"start_date {start_date} is after end_date {end_date}"
        )
    return start_date, end_date


@mcp.tool()
async def add_expense(
    date: str,
    amount: float,
    category: str,
    subcategory: str = "",
    note: str = "",
) -> dict:
    '''Add a new expense entry to the database. date must be YYYY-MM-DD.'''
    date = _check_date(date, "date")
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(
            "INSERT INTO expenses(date, amount, category, subcategory, note) VALUES (?,?,?,?,?)",
            (date, amount, category, subcategory, note)
        )
        expense_id = cur.lastrowid
        await c.commit()
        return {"status": "success", "id": expense_id, "message": "Expense added successfully"}


@mcp.tool()
async def list_expenses(start_date: str, end_date: str) -> list[dict]:
    '''List expense entries within an inclusive date range. Dates are YYYY-MM-DD.'''
    start_date, end_date = _check_range(start_date, end_date)
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC, id DESC
            """,
            (start_date, end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


@mcp.tool()
async def summarize(start_date: str, end_date: str, category: str | None = None) -> list[dict]:
    '''Summarize expenses by category within an inclusive date range. Dates are YYYY-MM-DD.'''
    start_date, end_date = _check_range(start_date, end_date)
    async with aiosqlite.connect(DB_PATH) as c:
        query = """
            SELECT LOWER(category) AS category, SUM(amount) AS total_amount, COUNT(*) as count
            FROM expenses
            WHERE date BETWEEN ? AND ?
        """
        params = [start_date, end_date]

        if category:
            query += " AND LOWER(category) = LOWER(?)"
            params.append(category)

        query += " GROUP BY LOWER(category) ORDER BY total_amount DESC"

        cur = await c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    # Read fresh each time so you can edit the file without restarting
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run(transport = 'http', host = "0.0.0.0", port = 8000)

import sqlite3

import pytest

from ejc_bot.stores.db import connect_readonly, describe_schema, run_select


def test_run_select_returns_dicts(schedule_db):
    rows = run_select(schedule_db, "SELECT title, location FROM events WHERE start_time = '14:00'")
    assert rows == [{"title": "Fight Night Qualifiers", "location": "Juggling Hall"}]


def test_run_select_caps_rows(schedule_db):
    assert len(run_select(schedule_db, "SELECT * FROM events", max_rows=2)) == 2


@pytest.mark.parametrize(
    "query",
    ["DELETE FROM events", "DROP TABLE events", "UPDATE events SET title = 'x'"],
)
def test_run_select_rejects_non_select(schedule_db, query):
    with pytest.raises(ValueError):
        run_select(schedule_db, query)


def test_write_through_cte_is_blocked_by_readonly_connection(schedule_db):
    with pytest.raises(sqlite3.OperationalError):
        run_select(schedule_db, "WITH x AS (SELECT 1) DELETE FROM events")


def test_stacked_statements_are_rejected(schedule_db):
    with pytest.raises(sqlite3.ProgrammingError):
        run_select(schedule_db, "SELECT 1; DROP TABLE events")
    assert len(run_select(schedule_db, "SELECT * FROM events")) == 3


def test_readonly_connection_cannot_write(schedule_db):
    conn = connect_readonly(schedule_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO events (title) VALUES ('x')")
    conn.close()


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect_readonly(tmp_path / "nope.db")


def test_describe_schema(schedule_db):
    assert describe_schema(schedule_db).startswith("id, title, day, start_time")

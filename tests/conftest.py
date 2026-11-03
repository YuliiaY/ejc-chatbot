import sqlite3

import pytest

from ejc_bot.config import get_settings
from ejc_bot.ingest.schedule import init_db
from ejc_bot.stores.vector import get_vector_store

SAMPLE_EVENTS = [
    ("KIDS SHOWS: Open Stage", "Tuesday, August 4th", "10:00", "13:00", "Vrazov trg"),
    ("Fight Night Qualifiers", "Tuesday, August 4th", "14:00", "17:00", "Juggling Hall"),
    ("Open-Air Cinema", "Tuesday, August 4th", "21:30", "23:30", "Panorama"),
]


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point every test at an empty temp data dir and reset cached singletons."""
    monkeypatch.setenv("EJC_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    get_vector_store.cache_clear()
    yield get_settings()
    get_settings.cache_clear()
    get_vector_store.cache_clear()


@pytest.fixture
def schedule_db(isolated_settings):
    path = isolated_settings.sql_db_path
    conn = sqlite3.connect(path)
    init_db(conn)
    conn.executemany(
        "INSERT INTO events (title, day, start_time, end_time, location) VALUES (?, ?, ?, ?, ?)",
        SAMPLE_EVENTS,
    )
    conn.commit()
    conn.close()
    return path

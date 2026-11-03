"""Scrape the EJC program page into the SQLite ``events`` table."""

import re
import sqlite3

import requests
from bs4 import BeautifulSoup

from ejc_bot.config import get_settings

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def init_db(conn):
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS events")
    cursor.execute("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            day TEXT,
            start_time TEXT,
            end_time TEXT,
            location TEXT,
            description TEXT,
            source_url TEXT
        )
    """)
    conn.commit()


def clean_text(text: str) -> str:
    """Strip extra whitespace and emojis from headings."""
    return re.sub(r"[^\w\s,:-]", "", text).strip()


def parse_time_and_title(raw_text: str):
    """Splits strings like '10:00 - 13:00 | KIDS SHOWS: ROSA GELATS (Vrazov trg)'
    into time components, title, and location."""
    time_str = ""
    start_time = ""
    end_time = ""
    title = raw_text
    location = ""

    # Check for Pipe separator '10:00 - 13:00 | Event Title'
    if "|" in raw_text:
        parts = raw_text.split("|", 1)
        time_str = parts[0].strip()
        title = parts[1].strip()
    else:
        # Fallback regex search for leading time formats (e.g., '11:00 Event Title')
        time_match = re.match(
            r"^((?:from\s+)?\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?)[\s|]+(.*)",
            raw_text,
            re.IGNORECASE,
        )
        if time_match:
            time_str = time_match.group(1).strip()
            title = time_match.group(2).strip()

    # Split start and end times if a range exists
    if "-" in time_str:
        times = time_str.split("-")
        start_time = times[0].replace("from", "").strip()
        end_time = times[1].strip()
    else:
        start_time = time_str.replace("from", "").strip()

    # Extract location if contained in trailing parentheses, e.g., (Vrazov trg)
    loc_match = re.search(r"\(([^)]+)\)$", title)
    if loc_match:
        location = loc_match.group(1).strip()
        title = title[: loc_match.start()].strip()

    return start_time, end_time, title, location


def scrape_and_insert():
    settings = get_settings()
    url = settings.program_url
    db_path = settings.sql_db_path

    print(f"Fetching program page: {url}")
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"Network error while fetching page: {e}.")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)
    cursor = conn.cursor()

    current_day = "General / Unscheduled"
    events_inserted = 0
    seen_events = set()  # Prevent duplicate insertions from repeated HTML elements

    for element in soup.find_all(["h3", "li"]):
        # Track the active day when hitting an <h3> tag
        if element.name == "h3":
            raw_heading = element.text
            if any(day_name in raw_heading.lower() for day_name in WEEKDAYS):
                current_day = clean_text(raw_heading)
            continue

        # Extract items under the active <h3> heading
        full_text = element.text.strip()
        if not full_text or full_text in seen_events:
            continue
        seen_events.add(full_text)

        start_time, end_time, title, location = parse_time_and_title(full_text)

        cursor.execute(
            """
            INSERT INTO events (title, day, start_time, end_time, location, source_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (title, current_day, start_time, end_time, location, url),
        )
        events_inserted += 1

    conn.commit()
    conn.close()
    print(f"Successfully inserted {events_inserted} program events into '{db_path}'.")


def main():
    scrape_and_insert()


if __name__ == "__main__":
    main()

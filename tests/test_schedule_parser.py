import pytest

from ejc_bot.ingest.schedule import clean_text, parse_time_and_title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "10:00 - 13:00 | KIDS SHOWS: ROSA GELATS (Vrazov trg)",
            ("10:00", "13:00", "KIDS SHOWS: ROSA GELATS", "Vrazov trg"),
        ),
        (
            "15:00 Markus Furtner show (Circus tent)",
            ("15:00", "", "Markus Furtner show", "Circus tent"),
        ),
        ("from 22:00 Fire Space", ("22:00", "", "Fire Space", "")),
        ("Tickets", ("", "", "Tickets", "")),
    ],
)
def test_parse_time_and_title(raw, expected):
    assert parse_time_and_title(raw) == expected


def test_clean_text_strips_emojis_but_keeps_day_heading():
    assert clean_text("🎪 Tuesday, August 4th ") == "Tuesday, August 4th"


def test_clean_text_removes_punctuation_outside_allowed_set():
    # The old pattern `[^\w\s,-:]` accidentally kept everything between ',' and ':' (e.g. '.', '/').
    assert clean_text("Mon. 3/8") == "Mon 38"

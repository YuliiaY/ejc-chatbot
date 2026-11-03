from ejc_bot.ingest.web import find_pdf_links


def test_relative_pdf_links_resolve_against_page_url():
    html = """
        <a href="docs/map.pdf">map</a>
        <a href="/terms.PDF">terms</a>
        <a href="/about/">about</a>
    """
    links = {"internal": [{"href": "https://ejc2026.org/info/rules.pdf"}], "external": []}

    found = find_pdf_links("https://ejc2026.org/info/page/", links, html)

    assert found == {
        "https://ejc2026.org/info/page/docs/map.pdf",
        "https://ejc2026.org/terms.PDF",
        "https://ejc2026.org/info/rules.pdf",
    }


def test_no_links():
    assert find_pdf_links("https://eja.net/", None, None) == set()

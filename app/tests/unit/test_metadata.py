"""Unit tests for filename → metadata extraction (no external services)."""

from datetime import date
from pathlib import Path

import pytest

from rag_system.ingest.metadata import extract_metadata, file_checksum


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Filename → company / date / version_label / doc_type
# ---------------------------------------------------------------------------
# `file_checksum` reads bytes, so we point at the assessment brief PDF that
# ships with the repo (small, always present). The same trick lets us write
# table-driven tests without spinning up Docling / Snowflake.
@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A trivial 'PDF' file — content doesn't matter for filename parsing."""
    p = tmp_path / "placeholder.pdf"
    p.write_bytes(b"%PDF-1.4\n%fake\n")
    return p


@pytest.mark.parametrize(
    "filename, exp_company, exp_ticker, exp_date, exp_version, exp_doc_type",
    [
        # Original corpus
        ("Digital Realty_Investor Presentation December 2025.pdf",
            "Digital Realty", "DLR", date(2025, 12, 1), "Dec 2025", "investor_presentation"),
        ("Digital Realty_Investor Presentation March 2026.pdf",
            "Digital Realty", "DLR", date(2026, 3, 1), "Mar 2026", "investor_presentation"),
        ("BXP Q4 2025 Investor Presentation with Appendix 3.20.2026.pdf",
            "Boston Properties", "BXP", date(2026, 3, 20), "Mar 2026", "investor_presentation"),
        ("EGP_2026_February_Roadshow_3.1_(Resize).pdf",
            "EastGroup Properties", "EGP", date(2026, 2, 1), "Feb 2026", "roadshow"),
        ("PSA Company-Update-Mar-26-vF.pdf",
            "Public Storage", "PSA", date(2026, 3, 1), "Mar 2026", "company_update"),
        ("PSA Merger-Presentation-vF.pdf",
            "Public Storage", "PSA", None, None, "merger_presentation"),
        ("Realty Incom q4-2025-investor-presentation.pdf",
            "Realty Income", "O", date(2025, 12, 1), "Dec 2025", "investor_presentation"),
        ("VICI-Investor-Presentation-Mar-26_.pdf",
            "VICI Properties", "VICI", date(2026, 3, 1), "Mar 2026", "investor_presentation"),
        ("BXP Morning Session Deck web.pdf",
            "Boston Properties", "BXP", None, None, "morning_session"),
        ("Simon The Impact of Brick and Mortar Shopping.pdf",
            "Simon Property Group", "SPG", None, None, "third_party_report"),
    ],
)
def test_extract_metadata_filename_cases(
    tmp_path: Path,
    filename: str,
    exp_company: str | None,
    exp_ticker: str | None,
    exp_date: date | None,
    exp_version: str | None,
    exp_doc_type: str | None,
) -> None:
    """The filename parser handles every shape in the corpus + a couple of edge cases."""
    target = tmp_path / filename
    target.write_bytes(b"%PDF-1.4\n")

    meta = extract_metadata(target)

    assert meta.company == exp_company, f"company mismatch for {filename}"
    assert meta.ticker == exp_ticker, f"ticker mismatch for {filename}"
    assert meta.doc_date == exp_date, f"doc_date mismatch for {filename}"
    assert meta.version_label == exp_version, f"version_label mismatch for {filename}"
    assert meta.doc_type == exp_doc_type, f"doc_type mismatch for {filename}"
    # checksum should always be a 64-hex-char sha256
    assert len(meta.checksum) == 64
    assert all(c in "0123456789abcdef" for c in meta.checksum)


def test_doc_id_is_stable_across_reads(fake_pdf: Path) -> None:
    """Same file → same doc_id (checksum drives the id suffix)."""
    a = extract_metadata(fake_pdf)
    b = extract_metadata(fake_pdf)
    assert a.doc_id == b.doc_id
    assert a.checksum == b.checksum


def test_doc_id_changes_when_content_changes(tmp_path: Path) -> None:
    """Two files with identical names but different content get different doc_ids."""
    a = tmp_path / "Digital Realty_Investor Presentation December 2025.pdf"
    b = tmp_path / "subdir" / "Digital Realty_Investor Presentation December 2025.pdf"
    b.parent.mkdir()
    a.write_bytes(b"%PDF-1.4\nfirst content\n")
    b.write_bytes(b"%PDF-1.4\nsecond content\n")

    meta_a = extract_metadata(a)
    meta_b = extract_metadata(b)
    assert meta_a.doc_id != meta_b.doc_id


def test_file_checksum_is_deterministic(fake_pdf: Path) -> None:
    assert file_checksum(fake_pdf) == file_checksum(fake_pdf)

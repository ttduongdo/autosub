"""Shared pytest fixtures for pipeline integration tests.

Two kinds of audio fixtures:
- sample_audio_path: a short, deterministic speech clip (known transcript),
  generated via macOS `say` the first time tests run. Fast, offline, exact
  ground truth — used for ingest/transcribe/align correctness tests.
- song_audio_path: a real song downloaded from SONG_TEST_URL (currently a
  SoundCloud track — YouTube's bot detection was blocking yt-dlp as of
  2026-10, so SoundCloud is used instead; swap platforms freely, this
  fixture works with whatever URL yt-dlp can resolve). Slow, needs network,
  no known ground truth — used for the "does this work on an actual song"
  integration test (stage 2 vocal separation, full-pipeline smoke test).
  Marked `@pytest.mark.song` and skipped by default; run explicitly with
  `pytest -m song`.

SONG_TEST_URL must be a URL you have the rights to use (your own upload, or
an explicitly CC-licensed/downloadable track) — see ARCHITECTURE.md's
data/rights notes.
"""
import subprocess
from pathlib import Path

import pytest

from pipeline.ingest import resolve_input

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

TEST_TRANSCRIPT = "hello world this is a test of the subtitle pipeline"
RAW_AIFF = FIXTURES_DIR / "sample.aiff"
SAMPLE_WAV = FIXTURES_DIR / "sample.wav"

SONG_TEST_URL = "https://soundcloud.com/ccottrill/sexy-to-someone-1"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "song: slow integration test that downloads a real song"
    )


@pytest.fixture(scope="session")
def sample_transcript() -> str:
    return TEST_TRANSCRIPT


@pytest.fixture(scope="session")
def sample_audio_path() -> Path:
    """Path to a short mono 16kHz WAV of sample_transcript being spoken.
    Generated once per test session via macOS `say` + ffmpeg, then cached
    on disk across runs.
    """
    if not SAMPLE_WAV.exists():
        subprocess.run(
            ["say", "-o", str(RAW_AIFF), TEST_TRANSCRIPT],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(RAW_AIFF),
                "-ac", "1",
                "-ar", "16000",
                str(SAMPLE_WAV),
            ],
            check=True,
            capture_output=True,
        )
    return SAMPLE_WAV


@pytest.fixture(scope="session")
def song_audio_path() -> Path:
    """Path to a normalized mono WAV downloaded from SONG_TEST_URL.
    Downloaded once per session via pipeline.ingest.resolve_input, then
    reused for the remainder of the test run (not cached across runs,
    since ingest downloads to data/raw, not tests/fixtures).
    """
    return resolve_input(SONG_TEST_URL)

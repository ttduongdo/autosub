"""Tests for finetune.dataset. Pure logic is tested directly; anything that
hits a live network (Billboard chart, lyrics.ovh, Jamendo API, YouTube) is
tested via mocking so this suite stays fast and doesn't require real
credentials or network access.

The one exception is test_build_manifest_against_live_billboard_source,
marked @pytest.mark.dataset and deselected by default (see pytest.ini) —
it hits real external sources and downloads real audio. Run explicitly
with: pytest -m dataset
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from finetune.dataset import (
    _clean_artist_name,
    _download_dali_audio,
    _fetch_lyrics,
    _filter_usable_jamendo_tracks,
    _split_by_artist,
    build_manifest,
)


# --- _filter_usable_jamendo_tracks ---

ENGLISH_LYRICS = (
    "I was walking down the street when I saw you standing there, "
    "and everything about that moment felt like it was meant to be."
)
SPANISH_LYRICS = (
    "Caminaba por la calle cuando te vi parada ahi, "
    "y todo sobre ese momento se sintio como si estuviera destinado a ser."
)


def test_filter_drops_tracks_without_lyrics():
    tracks = [
        {"id": "1", "lyrics": ENGLISH_LYRICS, "audiodownload": "http://example.com/1.mp3"},
        {"id": "2", "lyrics": "", "audiodownload": "http://example.com/2.mp3"},
        {"id": "3", "audiodownload": "http://example.com/3.mp3"},  # no lyrics key at all
    ]
    result = _filter_usable_jamendo_tracks(tracks)
    assert [t["id"] for t in result] == ["1"]


def test_filter_drops_tracks_without_download_url():
    tracks = [
        {"id": "1", "lyrics": ENGLISH_LYRICS, "audiodownload": "http://example.com/1.mp3"},
        {"id": "2", "lyrics": ENGLISH_LYRICS, "audiodownload": ""},
        {"id": "3", "lyrics": ENGLISH_LYRICS},  # no audiodownload key at all
    ]
    result = _filter_usable_jamendo_tracks(tracks)
    assert [t["id"] for t in result] == ["1"]


def test_filter_handles_whitespace_only_lyrics():
    tracks = [
        {"id": "1", "lyrics": "   \n  ", "audiodownload": "http://example.com/1.mp3"},
    ]
    assert _filter_usable_jamendo_tracks(tracks) == []


def test_filter_keeps_all_usable_english_tracks():
    tracks = [
        {"id": str(i), "lyrics": ENGLISH_LYRICS, "audiodownload": "http://example.com/x.mp3"}
        for i in range(5)
    ]
    assert len(_filter_usable_jamendo_tracks(tracks)) == 5


def test_filter_drops_non_english_lyrics():
    tracks = [
        {"id": "1", "lyrics": ENGLISH_LYRICS, "audiodownload": "http://example.com/1.mp3"},
        {"id": "2", "lyrics": SPANISH_LYRICS, "audiodownload": "http://example.com/2.mp3"},
    ]
    result = _filter_usable_jamendo_tracks(tracks)
    assert [t["id"] for t in result] == ["1"]


# --- _clean_artist_name ---

@pytest.mark.parametrize("raw,expected", [
    ("Karol G With Drake", "Karol G"),
    ("Ariana Grande", "Ariana Grande"),
    ("HARDY, Eric Church, Morgan Wallen & Tim McGraw", "HARDY"),
    ("Jhene Aiko Featuring Kendrick Lamar", "Jhene Aiko"),
    ("Ella Langley & Morgan Wallen", "Ella Langley"),
    ("  Spaced Out Name  ", "Spaced Out Name"),
])
def test_clean_artist_name(raw, expected):
    assert _clean_artist_name(raw) == expected


# --- _fetch_lyrics (mocked requests) ---

def test_fetch_lyrics_returns_text_on_success():
    mock_response = type("Resp", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: {"lyrics": "  some real lyrics here  "},
    })()
    with patch("finetune.dataset.requests.get", return_value=mock_response):
        result = _fetch_lyrics("Some Artist", "Some Title")
    assert result == "some real lyrics here"


def test_fetch_lyrics_returns_none_on_404():
    mock_response = type("Resp", (), {"status_code": 404})()
    with patch("finetune.dataset.requests.get", return_value=mock_response):
        result = _fetch_lyrics("Unknown Artist", "Unknown Title")
    assert result is None


def test_fetch_lyrics_returns_none_on_request_exception():
    with patch("finetune.dataset.requests.get", side_effect=requests.RequestException("timeout")):
        result = _fetch_lyrics("Some Artist", "Some Title")
    assert result is None


def test_fetch_lyrics_returns_none_on_empty_lyrics_field():
    mock_response = type("Resp", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: {"lyrics": "   "},
    })()
    with patch("finetune.dataset.requests.get", return_value=mock_response):
        result = _fetch_lyrics("Some Artist", "Some Title")
    assert result is None


# --- _split_by_artist ---

def test_split_keeps_each_artist_entirely_in_one_split():
    records = [
        {"song_id": f"song{i}", "artist": artist}
        for i, artist in enumerate(
            ["A", "A", "A", "B", "B", "C", "C", "C", "C", "D"]
        )
    ]
    result = _split_by_artist(records, eval_fraction=0.25, seed=0)

    artist_splits = {}
    for record in result:
        artist_splits.setdefault(record["artist"], set()).add(record["split"])

    for artist, splits in artist_splits.items():
        assert len(splits) == 1, f"artist {artist} appears in both splits: {splits}"


def test_split_produces_both_train_and_eval_with_enough_artists():
    records = [{"song_id": f"song{i}", "artist": f"artist{i}"} for i in range(10)]
    result = _split_by_artist(records, eval_fraction=0.2, seed=0)
    splits_present = {r["split"] for r in result}
    assert splits_present == {"train", "eval"}


def test_split_is_deterministic_given_same_seed():
    records = [
        {"song_id": f"song{i}", "artist": artist}
        for i, artist in enumerate(["A", "B", "C", "D", "E"])
    ]
    result_1 = _split_by_artist([r.copy() for r in records], seed=42)
    result_2 = _split_by_artist([r.copy() for r in records], seed=42)
    assert [r["split"] for r in result_1] == [r["split"] for r in result_2]


def test_split_handles_single_artist():
    records = [{"song_id": "song1", "artist": "OnlyArtist"}]
    result = _split_by_artist(records, eval_fraction=0.2)
    # with only one artist, it must land in exactly one split
    assert result[0]["split"] in ("train", "eval")


# --- _download_dali_audio (mocked resolve_input) ---

def test_download_dali_audio_returns_path_on_success():
    entry = {"song_id": "dali_1", "youtube_url": "https://www.youtube.com/watch?v=abc123"}
    fake_path = Path("/tmp/fake_audio.wav")

    with patch("finetune.dataset.resolve_input", return_value=fake_path) as mock_resolve:
        result = _download_dali_audio(entry, Path("/tmp"))

    mock_resolve.assert_called_once_with(entry["youtube_url"])
    assert result == fake_path


def test_download_dali_audio_returns_none_on_failure():
    entry = {"song_id": "dali_2", "youtube_url": "https://www.youtube.com/watch?v=dead"}

    with patch("finetune.dataset.resolve_input", side_effect=RuntimeError("blocked")):
        result = _download_dali_audio(entry, Path("/tmp"))

    assert result is None


# --- Live integration test (real Billboard chart + lyrics.ovh + YouTube + Demucs) ---

@pytest.mark.dataset
def test_build_manifest_against_live_billboard_source(tmp_path):
    output_path = tmp_path / "manifest.json"
    num_songs = 2

    result_path = build_manifest(num_songs=num_songs, output_path=output_path)

    assert result_path == output_path
    assert output_path.exists()

    records = json.loads(output_path.read_text())
    assert len(records) == num_songs

    for record in records:
        assert set(record.keys()) == {
            "song_id", "artist", "source", "audio_path", "vocals_path",
            "lyrics_text", "split",
        }
        assert record["source"] == "billboard"
        assert record["split"] in ("train", "eval")
        assert Path(record["audio_path"]).exists()
        assert Path(record["vocals_path"]).exists()
        assert record["lyrics_text"].strip() != ""

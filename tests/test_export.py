"""Tests for pipeline.export (stage 5). Pure local logic — fast, no models."""
import json

import pytest

from pipeline.export import _format_timestamp, write_srt, write_vtt, write_word_json


@pytest.fixture
def sample_segments():
    return [
        {"text": "hello world", "start": 1.5, "end": 4.2},
        {"text": "this is a test", "start": 4.2, "end": 7.8},
    ]


@pytest.fixture
def sample_words():
    return [
        {"word": "HELLO", "start": 1.5, "end": 2.0},
        {"word": "WORLD", "start": 2.1, "end": 2.6},
    ]


def test_format_timestamp_srt_style():
    assert _format_timestamp(1.5, ",") == "00:00:01,500"


def test_format_timestamp_vtt_style():
    assert _format_timestamp(1.5, ".") == "00:00:01.500"


def test_format_timestamp_handles_hours_and_minutes():
    # 1h 2m 3.25s
    seconds = 1 * 3600 + 2 * 60 + 3.25
    assert _format_timestamp(seconds, ",") == "01:02:03,250"


def test_write_srt_returns_path_and_creates_file(tmp_path, sample_segments):
    output_path = tmp_path / "out.srt"
    result = write_srt(sample_segments, output_path)

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text()
    assert "1\n00:00:01,500 --> 00:00:04,200\nhello world\n\n" in content
    assert "2\n00:00:04,200 --> 00:00:07,800\nthis is a test\n\n" in content


def test_write_srt_handles_none_end_time(tmp_path):
    segments = [{"text": "cut off segment", "start": 10.0, "end": None}]
    output_path = tmp_path / "out.srt"
    write_srt(segments, output_path)

    content = output_path.read_text()
    assert "00:00:10,000 --> 00:00:11,000" in content


def test_write_vtt_returns_path_and_has_header(tmp_path, sample_segments):
    output_path = tmp_path / "out.vtt"
    result = write_vtt(sample_segments, output_path)

    assert result == output_path
    content = output_path.read_text()
    assert content.startswith("WEBVTT\n\n")
    assert "00:00:01.500 --> 00:00:04.200" in content
    assert "," not in content.split("WEBVTT")[1]  # VTT uses periods, never commas


def test_write_vtt_handles_none_end_time(tmp_path):
    segments = [{"text": "cut off segment", "start": 10.0, "end": None}]
    output_path = tmp_path / "out.vtt"
    write_vtt(segments, output_path)

    content = output_path.read_text()
    assert "00:00:10.000 --> 00:00:11.000" in content


def test_write_word_json_round_trips(tmp_path, sample_words):
    output_path = tmp_path / "words.json"
    result = write_word_json(sample_words, output_path)

    assert result == output_path
    loaded = json.loads(output_path.read_text())
    assert loaded == sample_words

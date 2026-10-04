"""Build fine-tuning dataset manifest from lyric/audio pairs.

Primary source: Billboard Hot 100 (via a daily-updated public JSON mirror)
for song/artist names, LRCLIB for line-level timestamped (synced) lyrics,
and the existing pipeline.ingest.resolve_input (yt-dlp) for audio. This
intentionally prioritizes training-data relevance (recent, popular,
mainstream music — what the app will actually see in real use) over strict
licensing cleanliness, since this is a non-commercial portfolio/
demonstration project. See ARCHITECTURE.md's data/rights notes for the
fuller discussion.

Using LRCLIB specifically (rather than, say, running Whisper on each
training song to get segment timing) matters because it avoids a circular
dependency: using the model's own transcription to generate the timing
data used to train that same model would bake in whatever errors the
baseline already makes, rather than training against independently-correct
ground truth. LRCLIB's timestamps come from human-submitted synced lyrics
(the same format karaoke apps use), with no ASR model involved.

Each manifest record stores lyrics_segments (a list of {start, end, text}
line-level segments), not one lyrics_text blob — Whisper processes audio
in ~30s windows and caps decoder labels at 448 tokens, so a full song's
lyrics as one example doesn't fit. finetune/train.py groups these segments
into Whisper-sized chunks per song.

Secondary/fallback sources (optional use, not the primary path):
- Jamendo: Creative Commons tracks, audio + lyrics both available directly
  via Jamendo's API. See https://developer.jamendo.com/v3.0/tracks.
- DALI: purpose-built lyrics-alignment annotations, but audio is resolved
  via YouTube links via yt-dlp, which can fail (dead links, bot detection)
  — failures are skipped, not raised.

Splits the final manifest by artist (not song, not random chunks) into
train/eval to avoid leakage.
"""
import gzip
import json
import os
import random
import re
from pathlib import Path

import requests
import yt_dlp
from langdetect import DetectorFactory, LangDetectException, detect

from pipeline.separate import separate_vocals
from pipeline.ingest import _normalize_audio, resolve_input

# langdetect's n-gram model is probabilistic and non-deterministic by
# default (results can vary run-to-run on short/ambiguous text unless
# seeded) — fix the seed so classification is reproducible.
DetectorFactory.seed = 0

BILLBOARD_CHART_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/recent.json"
LRCLIB_API_URL = "https://lrclib.net/api/get"

JAMENDO_CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "")
JAMENDO_API_URL = "https://api.jamendo.com/v3.0/tracks"

RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "finetune"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# --- Billboard + LRCLIB source (primary) ---

LRC_LINE_RE = re.compile(r"^\[(\d{2}):(\d{2})(?:\.(\d{1,3}))?\](.*)$")


def _fetch_billboard_chart() -> list[dict]:
    """Fetch the current Billboard Hot 100 as a list of {song, artist, ...}
    dicts, from a daily-updated public JSON mirror (no API key needed).
    """
    response = requests.get(BILLBOARD_CHART_URL, timeout=30)
    response.raise_for_status()
    return response.json().get("data", [])


def _clean_artist_name(artist: str) -> str:
    """Strip collaborator/feature credits down to the primary artist name
    for lyrics lookups, which expect a single artist (e.g. "Karol G With
    Drake" -> "Karol G"). LRCLIB (like lyrics.ovh before it) matches best
    on just the primary artist, not the full collab credit string.
    """
    # split on the first occurrence of any of these separators
    for sep in (" Featuring ", " With ", " & ", ", "):
        if sep in artist:
            artist = artist.split(sep)[0]
    return artist.strip()


def _parse_lrc(lrc_text: str) -> list[dict]:
    """Parse LRC-format synced lyrics into [{start, end, text}, ...]. Each
    line's end time is the next line's start time (LRC only gives line
    start timestamps, not explicit end times); the last line's end is left
    as its start + a small fixed duration, since there's nothing to bound
    it against.
    """
    lines = []
    for raw_line in lrc_text.splitlines():
        match = LRC_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        minutes, seconds, millis, text = match.groups()
        text = text.strip()
        if not text:
            continue
        start = int(minutes) * 60 + int(seconds) + int((millis or "0").ljust(3, "0")) / 1000
        lines.append({"start": start, "text": text})

    segments = []
    for i, line in enumerate(lines):
        end = lines[i + 1]["start"] if i + 1 < len(lines) else line["start"] + 5.0
        segments.append({"start": line["start"], "end": end, "text": line["text"]})
    return segments


def _fetch_synced_lyrics(artist: str, title: str) -> list[dict] | None:
    """Query LRCLIB for a song's line-level synced lyrics. Returns None
    (never raises) if not found or no synced version exists — common,
    expected outcomes, not errors.

    Using LRCLIB instead of e.g. running Whisper on the song to get
    segment timing avoids a circular dependency: we'd otherwise be using
    the model's own (possibly wrong) output to generate the data used to
    train that same model. LRCLIB's timestamps are independently
    human-submitted, with no ASR involved.
    """
    try:
        response = requests.get(
            LRCLIB_API_URL,
            params={"artist_name": artist, "track_name": title},
            headers={"User-Agent": "AutoSub v1.0 (dataset builder)"},
            timeout=15,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        synced = data.get("syncedLyrics")
        if not synced:
            return None
        segments = _parse_lrc(synced)
        return segments or None
    except requests.RequestException as e:
        print(f"Skipping lyrics lookup for {artist!r} - {title!r}: {e}")
        return None


def _fetch_billboard_entries(limit: int) -> list[dict]:
    """Fetch the Billboard chart, look up synced lyrics for each entry, and
    keep only entries with real, English lyric segments. Returns dicts
    with song_id, artist, title, lyrics_segments — ready for audio
    resolution.
    """
    chart = _fetch_billboard_chart()
    entries = []
    for item in chart:
        if len(entries) >= limit:
            break
        title = item.get("song", "").strip()
        full_artist = item.get("artist", "").strip()
        if not title or not full_artist:
            continue

        primary_artist = _clean_artist_name(full_artist)
        segments = _fetch_synced_lyrics(primary_artist, title)
        if not segments:
            continue
        full_text = " ".join(s["text"] for s in segments)
        if not _is_english(full_text):
            continue

        entries.append({
            "song_id": f"billboard_{item.get('this_week', len(entries))}_{primary_artist}_{title}",
            "artist": primary_artist,
            "title": title,
            "lyrics_segments": segments,
        })
    return entries


def _search_and_download_audio(query: str, output_dir: Path) -> Path:
    """Search YouTube for `query` and download the best-match audio via
    yt-dlp's ytsearch1: scheme. Separate from pipeline.ingest.resolve_input,
    which only accepts an existing local file or a real http(s) URL — a
    search string doesn't fit that contract, so this calls yt_dlp directly.

    Re-enabling YouTube here (scoped to this dataset-building path only) is
    a deliberate, evidence-based call: ingest.py's docstring still notes
    YouTube downloads were unreliable earlier in this project (bot
    detection), but live testing during dataset-source research (2026-10)
    found 4/4 ytsearch1: downloads succeeded cleanly with no blocking.
    ingest.py's own URL-based path is left untouched.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(f"ytsearch1:{query}", download=True)
        # ytsearch wraps the single result in a playlist-style dict
        entry_info = info_dict["entries"][0] if "entries" in info_dict else info_dict
        return Path(ydl.prepare_filename(entry_info))


def _download_billboard_audio(entry: dict) -> Path | None:
    """Resolve audio for a Billboard entry via YouTube search + download,
    then normalize it the same way pipeline.ingest.resolve_input would.
    Returns None (never raises) on failure — a bad search match, no
    results, or a blocked/failed download are all expected possibilities,
    not errors worth crashing the whole manifest build over.
    """
    query = f"{entry['artist']} {entry['title']}"
    try:
        raw_path = _search_and_download_audio(query, RAW_DIR)
        return _normalize_audio(raw_path, sample_rate=16000)
    except Exception as e:
        print(f"Skipping Billboard entry {entry['song_id']}: {e}")
        return None


# --- Jamendo source (secondary/fallback) ---

def _fetch_jamendo_tracks(client_id: str = JAMENDO_CLIENT_ID, limit: int = 100, offset: int = 0) -> list[dict]:
    """Query Jamendo's /tracks endpoint with include=lyrics, lang=en,
    audiodownload_allowed=true, vocalinstrumental=vocal (excludes purely
    instrumental tracks, which can't have lyrics anyway). Returns raw track
    dicts from the API.

    Two things learned empirically while tuning this query:
    - include="lyrics+licenses" silently breaks the lyrics field — every
      track comes back with lyrics="" when "licenses" is combined with
      "lyrics" in the include param, even though the exact same tracks
      return real lyrics with include="lyrics" alone. We don't need
      license data for filtering right now, so include is just "lyrics".
    - lang=en initially looked broken (0 results) — but the real cause was
      Jamendo returning an HTTP 200 with headers.status="failed" (an
      "Internal Error"), which requests.raise_for_status() doesn't catch
      since it's a 200, not an HTTP error code. lang=en combined with
      order=listens_total does work and is checked for below. Without
      order=listens_total, lang=en consistently fails; this looks like a
      real Jamendo API quirk (certain parameter combinations error
      silently) rather than lang being unsupported.

    order=listens_total also matters independent of lang: popular tracks
    skew toward mainstream, English-dominant content, so sorting by it
    raises the usable-track hit rate substantially over default ordering
    (confirmed via manual testing: ~8% hit rate unfiltered vs ~50% with
    lang=en + order=listens_total together).
    """
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "include": "lyrics",
        "audiodownload_allowed": "true",
        "audioformat": "mp32",
        "vocalinstrumental": "vocal",
        "order": "listens_total",
        "lang": "en",
    }
    response = requests.get(JAMENDO_API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("headers", {}).get("status") != "success":
        raise RuntimeError(f"Jamendo API returned a non-success status: {data.get('headers')}")
    return data.get("results", [])


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        # too short or ambiguous to classify — treat as not confidently English
        return False


def _filter_usable_jamendo_tracks(tracks: list[dict]) -> list[dict]:
    """Drop tracks with empty/missing lyrics, non-English lyrics, or
    audiodownload_allowed=false."""
    usable = []
    for track in tracks:
        lyrics = (track.get("lyrics") or "").strip()
        if not lyrics:
            continue
        if not track.get("audiodownload"):
            continue
        if not _is_english(lyrics):
            continue
        usable.append(track)
    return usable


def _download_jamendo_audio(track: dict, output_dir: Path) -> Path:
    """Download one track's audio from its audiodownload URL."""
    dest = output_dir / f"jamendo_{track['id']}.mp3"
    if not dest.exists():
        response = requests.get(track["audiodownload"], timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


# --- DALI source (best-effort; audio depends on YouTube links) ---

def _load_dali_annotations(dali_data_dir: Path, limit: int) -> list[dict]:
    """Load DALI's .gz annotation files, extracting song_id, artist,
    line-level timed lyrics_segments, and the YouTube URL for each entry.

    NOTE: this schema (info/annotations/lines keys, each line's time as
    [start, end]) is based on DALI's documented structure but has not been
    verified against a real DALI .gz file — DALI access requires a
    separate request (see ARCHITECTURE.md), which hasn't come through yet
    as of this writing. Treat this as a best-effort scaffold to adjust once
    a real file is available to inspect.
    """
    entries = []
    for gz_path in sorted(dali_data_dir.glob("*.gz"))[:limit]:
        with gzip.open(gz_path, "rb") as f:
            data = json.loads(f.read())

        info = data.get("info", {})
        annotations = data.get("annotations", {})
        lines = annotations.get("lines", {})
        texts = lines.get("text", [])
        times = lines.get("time", [])

        segments = []
        for text, time_range in zip(texts, times):
            text = text.strip()
            if not text or not time_range or len(time_range) < 2:
                continue
            segments.append({"start": float(time_range[0]), "end": float(time_range[1]), "text": text})
        if not segments:
            continue

        entries.append({
            "song_id": f"dali_{gz_path.stem}",
            "artist": info.get("artist", "unknown"),
            "lyrics_segments": segments,
            "youtube_url": f"https://www.youtube.com/watch?v={info.get('id', '')}",
        })
    return entries


def _download_dali_audio(entry: dict, output_dir: Path) -> Path | None:
    """Attempt to download audio via yt-dlp for one DALI entry's YouTube
    URL. Returns None (never raises) on failure — dead link, blocked,
    etc. — so build_manifest can skip it and keep going.
    """
    try:
        return resolve_input(entry["youtube_url"])
    except Exception as e:
        print(f"Skipping DALI entry {entry['song_id']}: {e}")
        return None


# --- Split function ---

def _split_by_artist(records: list[dict], eval_fraction: float = 0.2, seed: int = 0) -> list[dict]:
    """Assign each record a 'split' of 'train' or 'eval', grouped so every
    song by the same artist lands in the same split — prevents the model
    being evaluated on an artist's voice it already saw during training.
    """
    artists = sorted({r["artist"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(artists)

    num_eval_artists = max(1, round(len(artists) * eval_fraction))
    eval_artists = set(artists[:num_eval_artists])

    for record in records:
        record["split"] = "eval" if record["artist"] in eval_artists else "train"
    return records

# --- Build dataset ---

def build_manifest(
    num_songs: int,
    output_path: Path,
    use_jamendo_fallback: bool = False,
    dali_data_dir: Path | None = None,
    max_jamendo_tracks_scanned: int = 1000,
) -> Path:
    """Build a fine-tuning dataset manifest, primarily from the current
    Billboard Hot 100 (song/artist via a public chart mirror, line-level
    timed lyrics via LRCLIB, audio via YouTube search). Optionally tops up
    with Jamendo and/or DALI if the primary source doesn't reach num_songs.

    Writes a JSON manifest: [{song_id, artist, source, audio_path,
    vocals_path, lyrics_segments, split}, ...], where lyrics_segments is a
    list of {start, end, text} line-level segments. Splits by artist into
    train/eval (never splits one artist's songs across both).

    Note: Jamendo's lyrics field is an unsegmented blob (no line timing
    available from that API), so Jamendo-sourced records get a single
    segment spanning the whole track — a real limitation versus the
    Billboard/LRCLIB and DALI paths, which have genuine line-level timing.
    """
    records = []

    billboard_entries = _fetch_billboard_entries(limit=num_songs)
    for entry in billboard_entries:
        if len(records) >= num_songs:
            break
        audio_path = _download_billboard_audio(entry)
        if audio_path is None:
            continue
        vocals_path, _instrumental_path, _mix_path = separate_vocals(audio_path)
        records.append({
            "song_id": entry["song_id"],
            "artist": entry["artist"],
            "source": "billboard",
            "audio_path": str(audio_path),
            "vocals_path": str(vocals_path),
            "lyrics_segments": entry["lyrics_segments"],
        })

    print(f"Billboard: found {len(records)}/{num_songs} usable tracks.")

    if use_jamendo_fallback and len(records) < num_songs:
        offset = 0
        scanned = 0
        while len(records) < num_songs and scanned < max_jamendo_tracks_scanned:
            tracks = _fetch_jamendo_tracks(limit=100, offset=offset)
            if not tracks:
                break
            offset += len(tracks)
            scanned += len(tracks)

            for track in _filter_usable_jamendo_tracks(tracks):
                if len(records) >= num_songs:
                    break
                audio_path = _download_jamendo_audio(track, RAW_DIR)
                vocals_path, _instrumental_path, _mix_path = separate_vocals(audio_path)
                records.append({
                    "song_id": f"jamendo_{track['id']}",
                    "artist": track.get("artist_name", "unknown"),
                    "source": "jamendo",
                    "audio_path": str(audio_path),
                    "vocals_path": str(vocals_path),
                    # Jamendo has no line-level timing — one segment for
                    # the whole track is the best available here.
                    "lyrics_segments": [{"start": 0.0, "end": track.get("duration", 30.0), "text": track["lyrics"]}],
                })

        if scanned:
            print(
                f"Jamendo: found {len(records)}/{num_songs} total usable "
                f"tracks after scanning {scanned} additional Jamendo tracks."
            )

    if dali_data_dir is not None and len(records) < num_songs:
        dali_entries = _load_dali_annotations(dali_data_dir, limit=num_songs - len(records))
        for entry in dali_entries:
            if len(records) >= num_songs:
                break
            audio_path = _download_dali_audio(entry, RAW_DIR)
            if audio_path is None:
                continue
            vocals_path, _instrumental_path, _mix_path = separate_vocals(audio_path)
            records.append({
                "song_id": entry["song_id"],
                "artist": entry["artist"],
                "source": "dali",
                "audio_path": str(audio_path),
                "vocals_path": str(vocals_path),
                "lyrics_segments": entry["lyrics_segments"],
            })

    records = _split_by_artist(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    return output_path

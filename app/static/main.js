const urlInput = document.getElementById("url-input");
const fileInput = document.getElementById("file-input");
const fileDropText = document.getElementById("file-drop-text");
const processButton = document.getElementById("process-button");
const statusLine = document.getElementById("status-line");
const srtLink = document.getElementById("srt-link");
const vttLink = document.getElementById("vtt-link");
const karaokeEmpty = document.getElementById("karaoke-empty");
const karaokeContent = document.getElementById("karaoke-content");
const waveformEl = document.getElementById("waveform");
const lyricsStrip = document.getElementById("lyrics-strip");
const stemToggle = document.getElementById("stem-toggle");
const playButton = document.getElementById("play-button");
const timeDisplay = document.getElementById("time-display");
const zoomSlider = document.getElementById("zoom-slider");

const STATUS_LABELS = {
  pending: "Starting…",
  downloading: "Downloading audio…",
  separating: "Separating vocals…",
  transcribing: "Transcribing…",
  aligning: "Aligning words…",
};

let wavesurfer = null;
let audioUrls = {};
let currentStem = "mix";
let currentSegments = [];
let currentWords = [];
let lyricsInner = null;
let wordEls = [];

fileInput.addEventListener("change", () => {
  fileDropText.textContent = fileInput.files.length
    ? fileInput.files[0].name
    : "Upload";
});

processButton.addEventListener("click", async () => {
  const url = urlInput.value.trim();
  const file = fileInput.files[0];

  if (!url && !file) {
    showStatus("Provide a URL or upload a file first.", true);
    return;
  }

  processButton.disabled = true;
  showStatus("Starting…", false);

  try {
    let jobId;
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      if (!res.ok) throw new Error(await res.text());
      jobId = (await res.json()).job_id;
    } else {
      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) throw new Error(await res.text());
      jobId = (await res.json()).job_id;
    }
    await pollJob(jobId);
  } catch (err) {
    showStatus(`Error: ${err.message}`, true);
    processButton.disabled = false;
  }
});

function showStatus(text, isError) {
  statusLine.hidden = false;
  statusLine.textContent = text;
  statusLine.classList.toggle("error", isError);
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/api/status/${jobId}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    if (data.status === "error") {
      showStatus(`Error: ${data.error}`, true);
      processButton.disabled = false;
      return;
    }

    if (data.status === "done") {
      statusLine.hidden = true;
      processButton.disabled = false;
      renderResult(data);
      return;
    }

    showStatus(STATUS_LABELS[data.status] || data.status, false);
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

function formatTime(seconds) {
  if (!isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function renderResult(data) {
  karaokeEmpty.hidden = true;
  karaokeContent.hidden = false;
  srtLink.href = data.srt_url;
  vttLink.href = data.vtt_url;
  audioUrls = data.audio_urls;
  currentStem = "mix";
  currentSegments = data.segments || [];
  currentWords = data.words || [];
  stemToggle.querySelectorAll(".stem-button").forEach((b) => b.classList.toggle("active", b.dataset.stem === "mix"));

  loadStem(audioUrls[currentStem], { seekTo: 0, autoplay: false });
}

function setupWaveform(src) {
  if (wavesurfer) {
    wavesurfer.destroy();
  }
  wavesurfer = WaveSurfer.create({
    container: waveformEl,
    waveColor: "#c3b595",
    progressColor: "#5c1f1a",
    cursorColor: "#5c1f1a",
    height: 128,
    barWidth: 2,
    barGap: 2,
    barRadius: 2,
    minPxPerSec: parseInt(zoomSlider.value, 10),
    url: src,
  });

  wavesurfer.on("timeupdate", onTimeUpdate);
  wavesurfer.on("play", () => (playButton.textContent = "⏸"));
  wavesurfer.on("pause", () => (playButton.textContent = "▶"));
  wavesurfer.on("ready", () => {
    timeDisplay.textContent = `0:00 / ${formatTime(wavesurfer.getDuration())}`;
    buildLyricsStrip(currentSegments, currentWords);
  });
  // wavesurfer v7's "scroll" event payload: (visibleStartTime, visibleEndTime,
  // scrollLeftPx, scrollWidthPx) — use the visible start time directly
  // rather than reaching into wavesurfer's internal (shadow) DOM for a
  // scroll offset, since its rendering internals aren't guaranteed stable.
  wavesurfer.on("scroll", (visibleStartTime) => {
    const pxPerSec = parseInt(zoomSlider.value, 10);
    lyricsStrip.scrollLeft = visibleStartTime * pxPerSec;
  });
  wavesurfer.on("zoom", () => buildLyricsStrip(currentSegments, currentWords));
}

async function loadStem(src, { seekTo: seekSeconds = 0, autoplay = false } = {}) {
  setupWaveform(src);
  wavesurfer.once("ready", () => {
    if (seekSeconds > 0) wavesurfer.setTime(seekSeconds);
    if (autoplay) wavesurfer.play();
  });
}

playButton.addEventListener("click", () => {
  if (wavesurfer) wavesurfer.playPause();
});

zoomSlider.addEventListener("input", () => {
  if (wavesurfer) wavesurfer.zoom(parseInt(zoomSlider.value, 10));
});

stemToggle.addEventListener("click", async (e) => {
  const button = e.target.closest(".stem-button");
  if (!button || !wavesurfer) return;
  const stem = button.dataset.stem;
  if (stem === currentStem) return;

  const wasPlaying = wavesurfer.isPlaying();
  const position = wavesurfer.getCurrentTime();

  stemToggle.querySelectorAll(".stem-button").forEach((b) => b.classList.remove("active"));
  button.classList.add("active");
  currentStem = stem;

  await loadStem(audioUrls[stem], { seekTo: position, autoplay: wasPlaying });
});

function seekTo(seconds) {
  if (!wavesurfer) return;
  const duration = wavesurfer.getDuration();
  if (duration > 0) {
    wavesurfer.seekTo(Math.min(seconds / duration, 1));
    wavesurfer.play();
  }
}

function buildLyricsStrip(segments, words) {
  lyricsStrip.innerHTML = "";
  wordEls = [];
  if (!segments || !segments.length || !wavesurfer) return;

  const duration = wavesurfer.getDuration();
  const pxPerSec = parseInt(zoomSlider.value, 10);
  const totalWidth = duration * pxPerSec;

  lyricsInner = document.createElement("div");
  lyricsInner.className = "lyrics-strip-inner";
  lyricsInner.style.width = `${totalWidth}px`;

  for (const w of words) {
    const left = w.start * pxPerSec;
    const block = document.createElement("div");
    block.className = "lyrics-word";
    block.textContent = w.word;
    block.dataset.start = w.start;
    block.dataset.end = w.end;
    block.title = `Jump to ${formatTime(w.start)}`;
    block.style.left = `${left}px`;
    block.addEventListener("click", () => seekTo(w.start));
    lyricsInner.appendChild(block);
    wordEls.push(block);
  }

  lyricsStrip.appendChild(lyricsInner);
}

function onTimeUpdate() {
  const t = wavesurfer.getCurrentTime();
  timeDisplay.textContent = `${formatTime(t)} / ${formatTime(wavesurfer.getDuration())}`;

  wordEls.forEach((w) => {
    const start = parseFloat(w.dataset.start);
    const end = parseFloat(w.dataset.end);
    w.classList.toggle("active", t >= start && t < end);
  });
}

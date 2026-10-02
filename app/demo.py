"""Gradio demo app wiring the pipeline together.

TODO:
- show baseline vs. fine-tuned WER comparison (needs a fine-tuned checkpoint)
"""
import base64
import uuid
from pathlib import Path

import gradio as gr

from pipeline.align import align_segments
from pipeline.export import write_srt, write_vtt
from pipeline.ingest import resolve_input
from pipeline.separate import separate_vocals
from pipeline.transcribe import transcribe

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MODEL = "openai/whisper-base"


KARAOKE_JS_ON_LOAD = """
() => {
  const container = document.querySelector("#karaoke-output");
  if (!container) return;
  const audio = container.querySelector("#player");
  const scroller = container.querySelector("#lyrics-scroll");
  const lines = container.querySelectorAll(".lyric-line");
  const words = container.querySelectorAll(".lyric-line span");
  if (!audio || audio.dataset.boundKaraoke) return;
  audio.dataset.boundKaraoke = "true";

  let activeLine = null;
  audio.addEventListener("timeupdate", () => {
    const t = audio.currentTime;
    words.forEach((w) => {
      const start = parseFloat(w.dataset.start);
      const end = parseFloat(w.dataset.end);
      w.classList.toggle("active", t >= start && t < end);
    });

    let currentLine = null;
    lines.forEach((line) => {
      const start = parseFloat(line.dataset.start);
      const end = parseFloat(line.dataset.end);
      const isActive = t >= start && t < end;
      line.classList.toggle("active-line", isActive);
      if (isActive) currentLine = line;
    });

    if (currentLine && currentLine !== activeLine) {
      activeLine = currentLine;
      scroller.scrollTo({
        top: currentLine.offsetTop - scroller.clientHeight / 2 + currentLine.clientHeight / 2,
        behavior: "smooth",
      });
    }
  });
}
"""


def _build_karaoke_html(audio_path: Path, segments: list[dict], words: list[dict]) -> str:
    """Self-contained HTML: an <audio> element (audio embedded as a base64
    data URI, so it works without any extra file-serving setup) plus one
    line per Whisper segment, each containing one <span> per aligned word.
    The highlighting + auto-scroll behavior is wired up separately via
    gr.HTML's js_on_load — <script> tags inside HTML handed to gr.HTML are
    inserted via innerHTML and never execute.
    """
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_src = f"data:audio/wav;base64,{audio_b64}"

    if not words or not segments:
        lines_html = "<em>(no word-level alignment available for this track)</em>"
    else:
        lines_html_parts = []
        for segment in segments:
            seg_start, seg_end = segment["start"], segment["end"] or segment["start"] + 1.0
            line_words = [w for w in words if seg_start <= w["start"] < seg_end]
            if not line_words:
                continue
            spans = " ".join(
                f'<span data-start="{w["start"]}" data-end="{w["end"]}">{w["word"]}</span>'
                for w in line_words
            )
            lines_html_parts.append(
                f'<div class="lyric-line" data-start="{seg_start}" data-end="{seg_end}">{spans}</div>'
            )
        lines_html = "\n".join(lines_html_parts)

    return f"""
    <div style="background: #141414; border-radius: 16px; padding: 28px; border: 1px solid #262626;">
      <audio id="player" controls style="width: 100%;" src="{audio_src}"></audio>
      <div id="lyrics-scroll" style="margin-top: 24px; height: 320px; overflow-y: auto; scroll-behavior: smooth; -webkit-mask-image: linear-gradient(to bottom, transparent, black 15%, black 85%, transparent); mask-image: linear-gradient(to bottom, transparent, black 15%, black 85%, transparent);">
        <div style="padding: 140px 0;">
          {lines_html}
        </div>
      </div>
    </div>
    <style>
      .lyric-line {{
        font-size: 1.4em;
        line-height: 2.3;
        color: #4a4a47;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        text-align: center;
        transition: color 0.3s ease;
        padding: 6px 0;
      }}
      .lyric-line.active-line {{ color: #8a8a85; }}
      .lyric-line span {{ padding: 2px 4px; border-radius: 6px; transition: color 0.15s ease; }}
      .lyric-line span.active {{ color: #f5f0e8; font-weight: 700; }}
    </style>
    """


def process(input_source: str, uploaded_file: str, model_choice: str):
    """Run the full pipeline on either a pasted URL or an uploaded file.

    Returns (karaoke_html, srt_path, vtt_path) for Gradio's outputs.
    """
    source = input_source.strip() if input_source and input_source.strip() else uploaded_file
    if not source:
        raise gr.Error("Provide a URL or upload a file first.")

    model_id = BASELINE_MODEL  # TODO: map model_choice to a fine-tuned checkpoint once one exists

    normalized_path = resolve_input(source)
    vocals_path, _mix_path = separate_vocals(normalized_path)
    segments = transcribe(vocals_path, model_id=model_id)
    words = align_segments(vocals_path, segments)

    run_id = uuid.uuid4().hex[:8]
    srt_path = write_srt(segments, OUTPUT_DIR / f"{run_id}.srt")
    vtt_path = write_vtt(segments, OUTPUT_DIR / f"{run_id}.vtt")

    karaoke_html = _build_karaoke_html(vocals_path, segments, words)
    return karaoke_html, str(srt_path), str(vtt_path)


CUSTOM_CSS = """
:root {
    --autosub-bg: #0a0a0a;
    --autosub-surface: #141414;
    --autosub-border: #262626;
    --autosub-text: #f5f5f0;
    --autosub-text-muted: #8a8a85;
    --autosub-accent: #f5f0e8;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: var(--autosub-bg) !important;
}

body, .gradio-container { color: var(--autosub-text) !important; }

#hero h1 {
    font-size: 2.1em;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 0.1em;
}
#hero p { color: var(--autosub-text-muted); font-size: 1.05em; }

#input-panel, #output-panel {
    background: var(--autosub-surface) !important;
    border: 1px solid var(--autosub-border) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}

#run-button {
    font-size: 1em;
    height: 48px;
    font-weight: 600;
    border-radius: 10px !important;
}

#karaoke-output { min-height: 320px; }

#main-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AutoSub") as demo:
        with gr.Column(elem_id="hero"):
            gr.Markdown(
                "# AutoSub\n"
                "Karaoke-style subtitles for any song or video — paste a link or upload a file."
            )

        with gr.Row(equal_height=False, elem_id="main-row"):
            with gr.Column(scale=2, elem_id="input-panel"):
                url_input = gr.Textbox(
                    label="URL",
                    placeholder="SoundCloud or any link yt-dlp supports (YouTube not supported right now)",
                )
                file_input = gr.File(
                    label="Or upload a file",
                    file_types=["audio", "video"],
                    type="filepath",
                )
                model_choice = gr.Radio(
                    choices=["Baseline Whisper"],  # TODO: add "Fine-tuned Whisper" once available
                    value="Baseline Whisper",
                    label="Model",
                )
                run_button = gr.Button("Process", variant="primary", elem_id="run-button")

                with gr.Row():
                    srt_output = gr.File(label="Download .srt")
                    vtt_output = gr.File(label="Download .vtt")

            with gr.Column(scale=3, elem_id="output-panel"):
                karaoke_output = gr.HTML(
                    label="Karaoke playback",
                    elem_id="karaoke-output",
                    js_on_load=KARAOKE_JS_ON_LOAD,
                )

        run_button.click(
            fn=process,
            inputs=[url_input, file_input, model_choice],
            outputs=[karaoke_output, srt_output, vtt_output],
        )

    return demo


if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0a0a0a",
        body_background_fill_dark="#0a0a0a",
        block_background_fill="#141414",
        block_background_fill_dark="#141414",
        block_border_color="#262626",
        block_border_color_dark="#262626",
        button_primary_background_fill="#f5f0e8",
        button_primary_background_fill_hover="#e8e2d6",
        button_primary_text_color="#0a0a0a",
        body_text_color="#f5f5f0",
        body_text_color_dark="#f5f5f0",
    )
    build_app().launch(theme=theme, css=CUSTOM_CSS)

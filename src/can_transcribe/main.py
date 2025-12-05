from pathlib import Path
from typing import Annotated

import torch
import typer

from .config import MOONSHINE_MODEL, PYANNOTE_MODEL
from .transcription import transcribe_exclusive_speakers

app = typer.Typer(
    name="can-transcribe",
    help="Simple audio transcription CLI that uses cutting edge models 🤖",
)


@app.command()
def transcribe(
    audio_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the audio or video file to transcribe",
            exists=True,
            readable=True,
        ),
    ],
    moonshine_model: Annotated[
        str,
        typer.Option("--model", "-m", help="Moonshine model to use for transcription"),
    ] = MOONSHINE_MODEL,
    diarization_model: Annotated[
        str,
        typer.Option("--diarization", "-d", help="Pyannote diarization model"),
    ] = PYANNOTE_MODEL,
    cpu: Annotated[
        bool,
        typer.Option("--cpu", help="Force CPU usage instead of CUDA"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug output"),
    ] = False,
) -> None:
    """Transcribe an audio or video file with speaker diarization."""
    if not cpu and not torch.cuda.is_available():
        typer.echo("⚠️  CUDA not available, falling back to CPU", err=True)
        cpu = True

    device = "cpu" if cpu else "cuda"

    transcribe_exclusive_speakers(
        audio_file=str(audio_file),
        moonshine_model=moonshine_model,
        diarization_model=diarization_model,
        device=device,
        debug=debug,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from .config import MOONSHINE_MODEL, PYANNOTE_MODEL
from .transcription import transcribe_exclusive_speakers

app = typer.Typer(
    name="can-transcribe",
    help="Simple audio transcription CLI that uses cutting edge models 🤖",
)

console = Console()


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
    visual: Annotated[
        bool,
        typer.Option("--visual", "-v", help="Enable visual analysis (face ID, emotion, scene descriptions)"),
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

    if visual:
        # Check if input is a video file
        from .config import VIDEO_EXTENSIONS
        if audio_file.suffix.lower() not in VIDEO_EXTENSIONS:
            typer.echo("⚠️  Visual analysis requires a video file", err=True)
            raise typer.Exit(1)

        from .visual_transcription import transcribe_with_vision
        transcribe_with_vision(
            video_file=str(audio_file),
            moonshine_model=moonshine_model,
            diarization_model=diarization_model,
            device=device,
            debug=debug,
        )
    else:
        transcribe_exclusive_speakers(
            audio_file=str(audio_file),
            moonshine_model=moonshine_model,
            diarization_model=diarization_model,
            device=device,
            debug=debug,
        )


def main() -> None:
    app()


@app.command()
def check(
    video_file: Annotated[
        Optional[Path],
        typer.Argument(
            help="Optional video file to test hardware-accelerated decoding",
            exists=True,
            readable=True,
        ),
    ] = None,
) -> None:
    """Run diagnostic checks on system components."""
    import shutil
    import subprocess

    table = Table(title="🔍 System Diagnostics")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    all_ok = True

    # --- Python & PyTorch ---
    import sys
    table.add_row("Python", "✅", sys.version.split()[0])

    torch_version = torch.__version__
    table.add_row("PyTorch", "✅", torch_version)

    # --- CUDA ---
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability()
        cuda_version = torch.version.cuda or "N/A"
        table.add_row("CUDA", "✅", f"{cuda_version} ({gpu_name}, CC {major}.{minor})")
    else:
        table.add_row("CUDA", "⚠️", "Not available (CPU mode)")
        all_ok = False

    # --- Flash Attention ---
    from .config import supports_flash_attention
    if supports_flash_attention():
        try:
            import flash_attn
            fa_version = getattr(flash_attn, "__version__", "unknown")
            table.add_row("Flash Attention", "✅", f"v{fa_version}")
        except ImportError:
            table.add_row("Flash Attention", "❌", "Import failed")
    else:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                table.add_row("Flash Attention", "⚠️", f"Requires CC 8.0+ (have {major}.x)")
            else:
                table.add_row("Flash Attention", "⚠️", "Not installed (optional)")
        else:
            table.add_row("Flash Attention", "⚠️", "Requires CUDA")

    # --- FFmpeg ---
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            table.add_row("FFmpeg", "✅", version_line.replace("ffmpeg version ", "")[:50])
        except Exception as e:
            table.add_row("FFmpeg", "⚠️", str(e))
    else:
        table.add_row("FFmpeg", "❌", "Not found in PATH")
        all_ok = False

    # --- FFmpeg Hardware Acceleration ---
    hwaccels = []
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Skip header line "Hardware acceleration methods:"
            hwaccels = [line.strip() for line in lines[1:] if line.strip()]
    except Exception:
        pass

    if hwaccels:
        table.add_row("FFmpeg HW Accel", "✅", ", ".join(hwaccels))
    else:
        table.add_row("FFmpeg HW Accel", "⚠️", "None detected")

    # --- FFmpeg Decoders (AV1, HEVC, H264) ---
    # Note: NVIDIA uses cuvid suffix, not nvdec for decoder names
    decoders_to_check = {
        "av1": ["av1_cuvid", "av1_vaapi", "av1_qsv", "libdav1d", "libaom-av1", "av1"],
        "hevc": ["hevc_cuvid", "hevc_vaapi", "hevc_qsv", "hevc"],
        "h264": ["h264_cuvid", "h264_vaapi", "h264_qsv", "h264"],
    }

    try:
        result = subprocess.run(
            ["ffmpeg", "-decoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        available_decoders = result.stdout if result.returncode == 0 else ""

        for codec_name, decoder_list in decoders_to_check.items():
            found = [d for d in decoder_list if f" {d} " in available_decoders or f" {d}\n" in available_decoders]
            if found:
                hw_decoders = [d for d in found if any(x in d for x in ["nvdec", "vaapi", "qsv", "cuvid"])]
                if hw_decoders:
                    table.add_row(f"{codec_name.upper()} Decode", "✅", f"HW: {', '.join(hw_decoders)}")
                else:
                    table.add_row(f"{codec_name.upper()} Decode", "✅", f"SW: {', '.join(found)}")
            else:
                table.add_row(f"{codec_name.upper()} Decode", "⚠️", "Not available")
    except Exception as e:
        table.add_row("Decoder Check", "⚠️", str(e))

    # --- Hugging Face Token ---
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            table.add_row("HuggingFace Token", "✅", f"Set ({token[:8]}...)")
        else:
            table.add_row("HuggingFace Token", "⚠️", "Not set (needed for pyannote)")
    except Exception:
        table.add_row("HuggingFace Token", "⚠️", "Could not check")

    # --- Core Dependencies ---
    # Check Moonshine via transformers
    try:
        from transformers import MoonshineForConditionalGeneration
        table.add_row("Moonshine (ASR)", "✅", "via transformers")
    except ImportError:
        table.add_row("Moonshine (ASR)", "❌", "MoonshineForConditionalGeneration not found")
        all_ok = False

    core_deps = [
        ("pyannote.audio", "Pyannote Audio"),
        ("transformers", "Transformers"),
    ]
    for module, name in core_deps:
        try:
            mod = __import__(module.split(".")[0])
            version = getattr(mod, "__version__", "installed")
            table.add_row(name, "✅", version)
        except ImportError:
            table.add_row(name, "❌", "Not installed")
            all_ok = False

    # --- Vision Dependencies (optional) ---
    vision_deps = [
        ("insightface", "InsightFace"),
        ("onnxruntime", "ONNX Runtime"),
        ("scenedetect", "PySceneDetect"),
    ]
    for module, name in vision_deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            table.add_row(name, "✅", version)
        except ImportError:
            table.add_row(name, "⚠️", "Not installed (visual mode)")

    console.print(table)

    # --- Video File Test ---
    if video_file:
        console.print("\n[bold]🎬 Testing video decoding...[/bold]")

        # Get video codec info
        try:
            probe_result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,duration",
                    "-of", "csv=p=0",
                    str(video_file),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe_result.returncode == 0:
                info = probe_result.stdout.strip()
                console.print(f"  Video info: [cyan]{info}[/cyan]")
        except Exception as e:
            console.print(f"  [yellow]Could not probe video: {e}[/yellow]")

        # Test hardware-accelerated decode
        try:
            test_result = subprocess.run(
                [
                    "ffmpeg",
                    "-hwaccel", "auto",
                    "-i", str(video_file),
                    "-t", "1",  # Only decode 1 second
                    "-f", "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            stderr = test_result.stderr or ""

            # Check for hardware acceleration usage
            hw_used = None
            for hw in ["cuda", "nvdec", "vaapi", "qsv", "videotoolbox", "d3d11va"]:
                if hw in stderr.lower():
                    hw_used = hw
                    break

            if test_result.returncode == 0:
                if hw_used:
                    console.print(f"  [green]✅ Hardware decode successful ({hw_used})[/green]")
                else:
                    console.print("  [green]✅ Decode successful (software)[/green]")
            else:
                console.print(f"  [red]❌ Decode failed[/red]")
                # Show relevant error
                for line in stderr.split("\n"):
                    if "error" in line.lower() or "failed" in line.lower():
                        console.print(f"     [dim]{line.strip()}[/dim]")
                all_ok = False

        except subprocess.TimeoutExpired:
            console.print("  [yellow]⚠️ Decode test timed out[/yellow]")
        except Exception as e:
            console.print(f"  [red]❌ Error: {e}[/red]")
            all_ok = False

    # --- Summary ---
    console.print()
    if all_ok:
        console.print("[bold green]✅ All checks passed![/bold green]")
    else:
        console.print("[bold yellow]⚠️ Some checks failed. See above for details.[/bold yellow]")


if __name__ == "__main__":
    main()

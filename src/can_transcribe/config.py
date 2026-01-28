# --- CONFIGURATION ---
import torch

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

MOONSHINE_MODEL = "UsefulSensors/moonshine-tiny"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
QWEN3_VL_MODEL = "Qwen/Qwen3-VL-2B-Thinking"


def supports_flash_attention() -> bool:
    """
    Auto-detect if Flash Attention 2 is supported.

    Requirements:
    - CUDA GPU available
    - Compute Capability 8.0+ (Ampere: RTX 30xx, 40xx, A100, H100)
    - flash_attn library installed

    Returns:
        True if Flash Attention 2 can be used, False otherwise.
    """
    # Check if we have a GPU
    if not torch.cuda.is_available():
        return False

    # Check GPU architecture (Ampere or newer required)
    # Flash Attn 2 requires Compute Capability 8.0+
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False

    # Check if the library is actually installed
    try:
        import flash_attn  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def get_device() -> str:
    """
    Auto-detect best available compute device.

    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

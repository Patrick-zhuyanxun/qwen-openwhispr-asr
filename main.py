import argparse
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Annotated

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from qwen_asr import Qwen3ASRModel


DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
MODEL_BY_SIZE = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}

app = FastAPI(title="Qwen3-ASR OpenWhispr Bridge")

_model: Qwen3ASRModel | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _torch_dtype(name: str) -> torch.dtype:
    normalized = (name or "auto").strip().lower()
    if normalized == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported QWEN_ASR_DTYPE={name!r}")


def _device_map() -> str:
    configured = os.getenv("QWEN_ASR_DEVICE", "auto").strip().lower()
    if configured and configured != "auto":
        return configured
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _model_name(requested_model: str | None = None) -> str:
    requested = (requested_model or "").strip()
    if requested and (requested.startswith("Qwen/") or Path(requested).exists()):
        return requested
    return os.getenv("QWEN_ASR_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _model_from_size(model_size: str | None) -> str:
    normalized = (model_size or "").strip().upper()
    if normalized in MODEL_BY_SIZE:
        return MODEL_BY_SIZE[normalized]
    raise ValueError(f"Unsupported model size: {model_size!r}. Use 0.6B or 1.7B.")


def load_model(requested_model: str | None = None) -> Qwen3ASRModel:
    global _model
    with _model_lock:
        if _model is not None:
            return _model

        dtype = _torch_dtype(os.getenv("QWEN_ASR_DTYPE", "auto"))
        max_new_tokens = int(os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "512"))
        model_id = _model_name(requested_model)
        device_map = _device_map()

        print(
            f"Loading Qwen3-ASR model={model_id} device_map={device_map} dtype={dtype}",
            flush=True,
        )
        _model = Qwen3ASRModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=1,
        )
        return _model


def _normalize_language(language: str | None) -> str | None:
    value = (language or "").strip()
    if not value or value.lower() == "auto":
        return None

    aliases = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "zh-hant": "Chinese",
        "zh-hans": "Chinese",
        "cn": "Chinese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
        "ja": "Japanese",
        "jp": "Japanese",
        "ko": "Korean",
        "kr": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
        "vi": "Vietnamese",
        "th": "Thai",
        "id": "Indonesian",
        "ms": "Malay",
        "tr": "Turkish",
        "hi": "Hindi",
        "ar": "Arabic",
    }
    return aliases.get(value.lower(), value[:1].upper() + value[1:])


def _convert_to_wav(input_path: Path, output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg is required but was not found")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"ffmpeg could not decode audio: {completed.stderr.strip()}",
        )


def _transcribe_file(
    audio_path: Path,
    model_name: str,
    language: str | None,
    prompt: str | None,
) -> str:
    model = load_model(model_name)
    normalized_language = _normalize_language(language)
    context = prompt or ""

    with _inference_lock:
        result = model.transcribe(
            str(audio_path),
            context=context,
            language=normalized_language,
            return_time_stamps=False,
        )

    return result[0].text.strip() if result else ""


async def _handle_transcription(
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
):
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="qwen-openwhispr-") as tmp:
        raw_path = Path(tmp) / f"input{suffix}"
        wav_path = Path(tmp) / "input.wav"
        raw_path.write_bytes(await file.read())
        _convert_to_wav(raw_path, wav_path)
        text = _transcribe_file(wav_path, model, language, prompt)

    if (response_format or "").strip().lower() == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": os.getenv("QWEN_ASR_MODEL", DEFAULT_MODEL),
        "loaded": _model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/v1/models")
def models():
    model_id = os.getenv("QWEN_ASR_MODEL", DEFAULT_MODEL)
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = DEFAULT_MODEL,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
):
    return await _handle_transcription(file, model, language, prompt, response_format)


@app.post("/audio/transcriptions")
async def openai_transcriptions_without_v1(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = DEFAULT_MODEL,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
):
    return await _handle_transcription(file, model, language, prompt, response_format)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenWhispr-compatible Qwen3-ASR server.")
    parser.add_argument("--host", default=os.getenv("QWEN_ASR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QWEN_ASR_PORT", "8179")))
    parser.add_argument(
        "--model-size",
        default=os.getenv("QWEN_ASR_MODEL_SIZE", "0.6B"),
        help="Qwen3-ASR model size to load: 0.6B or 1.7B. Defaults to 0.6B.",
    )
    parser.add_argument("--warmup", action="store_true", help="Load the ASR model before serving.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["QWEN_ASR_MODEL"] = _model_from_size(args.model_size)
    if args.warmup:
        load_model()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

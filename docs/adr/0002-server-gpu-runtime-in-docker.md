# ADR 0002 — Server GPU runtime in Docker

- **Status:** Accepted
- **Date:** 2026-10-16
- **Issue:** #2 (spike, time-boxed to 1 day)

## Context

The Parlora server is planned to run in Docker with GPU access on Windows
(Docker Desktop, WSL2 backend). The plan is a slim Python 3.12 image with the
CUDA user-space libraries (cuBLAS, cuDNN) installed as pip wheels, instead of
a multi-GB `nvidia/cuda` base image. Speech-to-text will use faster-whisper
`large-v3-turbo` with int8 weights on an RTX 4060 (8 GB VRAM).

The local-runtime work (Docker Compose, server image) builds on this choice,
so it had to be checked on real hardware first.

## Questions

1. Does a Compose GPU reservation (or `--gpus all`) expose the RTX 4060 inside
   a `python:3.12-slim`-based container on Docker Desktop/WSL2?
2. Does faster-whisper `large-v3-turbo` int8 load on CUDA with only the
   `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels?
3. How much VRAM does it use, and how long does a 10-second synthetic clip
   take to transcribe — container vs native?
4. What are the image size and cold-start time?

## Setup

| Item        | Value                                                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Host        | Windows 11 Pro, RTX 4060 8 GB, NVIDIA driver 616.56, 16 GB RAM                                                                            |
| Docker      | Docker Desktop, engine 28.2.2, WSL2 backend                                                                                               |
| Base image  | `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` (= `python:3.12-slim` + uv)                                                               |
| Packages    | faster-whisper 1.2.1, ctranslate2 4.8.3, nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 9.27.0.42 (pulls nvidia-cuda-nvrtc-cu12 12.9.86) |
| Model       | `large-v3-turbo`, `compute_type="int8"`, `device="cuda"`, beam size 5                                                                     |
| Audio       | 11.35 s of synthetic speech (Windows SAPI voice, 16 kHz mono); no real voice used, nothing committed                                      |
| Model cache | Named Docker volume mounted as `HF_HOME`                                                                                                  |
| User        | Non-root (uid 10001)                                                                                                                      |

The spike code (Dockerfile, measurement script, Compose file) stayed on a
throwaway branch and is not merged.

## Findings

### 1. GPU in a slim container — yes

- `docker run --gpus all` and a Compose service with
  `deploy.resources.reservations.devices: [{capabilities: [gpu]}]` both expose
  the RTX 4060.
- `nvidia-smi` works inside the container without being installed in the
  image; Docker Desktop injects it (`/usr/bin/nvidia-smi`) together with the
  driver libraries.

### 2. CUDA libraries as pip wheels — yes, with one required setting

- The model loads and runs on CUDA with only the cuBLAS and cuDNN wheels.
- ctranslate2 does **not** find the wheel libraries by itself. Without
  `LD_LIBRARY_PATH` pointing at `site-packages/nvidia/cublas/lib` and
  `site-packages/nvidia/cudnn/lib`, it fails with
  `RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`.
- faster-whisper 1.2.1 is **incompatible with PyAV 19** (the latest at the
  time): its file decoder passes `metadata_errors` to `av.open()`, which PyAV
  19 no longer accepts. PyAV 14 has no wheel for this image and falls back to
  a source build that fails. Passing in-memory float32 PCM to
  `model.transcribe()` avoids PyAV entirely. That is also what the voice
  pipeline will do, since it receives audio frames, not files.

### 3. VRAM and speed — well within budget

Two container runs, model already cached, 5 warm runs each:

| Metric                               | Run 1     | Run 2     |
| ------------------------------------ | --------- | --------- |
| VRAM after model load (delta)        | 1,079 MiB | 1,085 MiB |
| VRAM peak while transcribing (delta) | 1,161 MiB | 1,167 MiB |
| First transcription (warm-up)        | 2.53 s    | 1.61 s    |
| Warm transcription, median of 5      | 0.805 s   | 0.747 s   |
| Real-time factor (median)            | 0.071     | 0.066     |
| Word error rate vs script            | 0.0       | 0.0       |

Speech-to-text needs about 1.2 GB of the 8 GB VRAM, which leaves room for the
LLM and TTS.

**Native comparison: not measured.** The project keeps the toolchain off the
host (Dev Container decision), so there was no native Python to compare
against inside the timebox. Container overhead on GPU compute is expected to
be small, but this is unverified. The comparison moves to the follow-up change
`add-component-benchmarks`.

### 4. Image size and cold start

| Metric                                                              | Value                                                      |
| ------------------------------------------------------------------- | ---------------------------------------------------------- |
| Compressed image (registry/pull size)                               | 1.66 GB                                                    |
| `site-packages` in the image                                        | 2.6 GB, of which cuDNN 1.2 GB, cuBLAS 0.8 GB, NVRTC 0.2 GB |
| Size shown by Docker Desktop (compressed + unpacked)                | 4.65 GB                                                    |
| First build, base image not cached                                  | 271 s                                                      |
| Model download (first run, `large-v3-turbo`)                        | one-off, into the named volume                             |
| Cold start, `docker run` until the result is printed (model cached) | 15.5–19.1 s                                                |
| — Python imports                                                    | 1.1 s                                                      |
| — Model load to GPU                                                 | 7.0–8.4 s                                                  |
| — First transcription (CUDA warm-up)                                | 1.6–2.5 s                                                  |

The CUDA wheels make up about 80 % of the image. The model weights are not in
the image; they live in a volume.

## Decision

Keep design decision D12 as planned:

- The server image is based on the official uv Python 3.12 slim image. No
  `nvidia/cuda` base image.
- The GPU is reserved in `compose.yaml` with
  `deploy.resources.reservations.devices: [{capabilities: [gpu]}]`; a `nogpu`
  override removes it.
- CUDA user-space libraries arrive as pip wheels **in Phase 1**, together with
  faster-whisper, not in the skeleton. The skeleton image stays small.

Requirements for the Phase 1 image, from these findings:

1. Set `LD_LIBRARY_PATH` to the cuBLAS and cuDNN wheel library directories.
2. Feed faster-whisper in-memory PCM arrays; do not depend on its PyAV file
   decoder. If file decoding is ever needed, pin a PyAV version that
   faster-whisper supports and that ships a wheel.
3. Store model weights in a named volume (`HF_HOME`), never in the image.
4. Load the model once at server start and run one warm-up transcription
   before reporting ready, so the first learner turn does not pay the 1.6–2.5 s
   CUDA warm-up.
5. Use a multi-stage build so the CUDA wheel layer is cached separately from
   application code.

## Consequences

- The local-runtime work can go ahead without changes to its design or tasks.
- The Phase 1 server image will be about 1.7 GB compressed (about 2.8 GB on
  disk) because of the CUDA wheels. Faster rebuilds depend on layer caching.
- Server readiness takes about 10–12 s after container start (imports, model
  load, warm-up). The readiness check must allow for that.
- The latency budget for speech-to-text is about 0.8 s per 11 s utterance on
  the target GPU, and shorter turns will be faster.
- The native-versus-container comparison remains open and is tracked in
  `add-component-benchmarks`.

## Options ruled out

- **`nvidia/cuda:*-cudnn-runtime` base image.** Not needed: the slim image
  plus wheels works. A CUDA base image would add size and couple the image to
  one CUDA release without solving anything this spike found.
- **Running the server natively on Windows.** Not needed for GPU access. It
  would also break the Dev Container decision to keep the toolchain off the
  host.

## Amendment (2026-11-24): one cuDNN for CUDA 13

Phase 1 added onnxruntime-gpu for Kokoro. Its CUDA 13 build brings
`nvidia-cudnn-cu13`, and the server also asked for `nvidia-cudnn-cu12` for
CTranslate2. Both wheels write the same files under `nvidia/cudnn`, so the
installer's order decided which build each library came from. The image ended
up with a mix that could not load cuDNN's runtime-compiled engines
(`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`), and onnxruntime moved Kokoro to the
CPU in the middle of the warm-up: about 1–2.7 s to the first audio instead of
about 0.25 s, in a real call.

- The server installs **one cuDNN: `nvidia-cudnn-cu13`**, from onnxruntime-gpu.
  CTranslate2 (cuBLAS for CUDA 12, `nvidia-cublas-cu12`) runs on it; this is
  the combination the native Windows benchmarks already used.
- `LD_LIBRARY_PATH` also lists `nvidia/cu13/lib`, where cuDNN finds NVRTC 13
  at run time.
- A repository test fails if the lock holds more than one cuDNN wheel, and
  Kokoro logs a warning when onnxruntime leaves CUDA during inference.
- An existing environment that removes `nvidia-cudnn-cu12` also loses the
  shared files; reinstall the CUDA 13 wheel afterwards
  (`uv sync --reinstall-package nvidia-cudnn-cu13`).

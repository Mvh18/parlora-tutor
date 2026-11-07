# ADR 0009 — Phase 1 LLM, VRAM budget and latency from the first benchmarks

- **Status:** Accepted
- **Date:** 2026-11-07
- **Issue:** #69 (epic #61, component benchmarks)

## Context

Phase 1 builds the minimal voice loop: voice activity detection, speech-to-text,
LLM and text-to-speech on one machine with an RTX 4060 (8 GB). Before it starts it
needs a default LLM, a VRAM budget per component and an answer to ADR 0002's open
question: does speech-to-text run natively as well as in the container? The
component benchmarks (`scripts/bench/`, report in
[docs/benchmarks.md](../benchmarks.md)) measured each component on its own and the
GPU with everything loaded. The numbers below come from these results:

- LLM: [qwen3:4b-instruct](../../benchmarks/results/20261106T173349Z-llm-qwen3-4b-instruct-2507-q4-k-m.json),
  [gemma3:4b](../../benchmarks/results/20261106T173409Z-llm-gemma3-4b.json),
  [qwen3:8b](../../benchmarks/results/20261106T173430Z-llm-qwen3-8b.json)
- Speech-to-text: [native](../../benchmarks/results/20261106T175745Z-stt-native.json),
  [container](../../benchmarks/results/20261106T180314Z-stt-container.json)
- Text-to-speech: [native](../../benchmarks/results/20261106T180738Z-tts-native.json),
  [container](../../benchmarks/results/20261106T180934Z-tts-container.json)
- Memory: [STT and each LLM together](../../benchmarks/results/20261107T010309Z-memory-native.json)

Measured (warm unless noted; VRAM above a baseline of about 1 GB for the desktop):

| Component                            | Result                                                                                             |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `qwen3:4b-instruct-2507-q4_K_M`      | TTFT p50 0.045 s, p95 0.100 s; 77.5 tokens/s; 3.1 GB                                               |
| `gemma3:4b`                          | TTFT p50 0.064 s, p95 0.079 s; 76.4 tokens/s; 3.7 GB                                               |
| `qwen3:8b`                           | TTFT p50 0.062 s, p95 0.086 s; 45.6 tokens/s; 5.4 GB                                               |
| faster-whisper `large-v3-turbo` int8 | p50 0.41 s, p95 0.49 s native (0.43 / 0.55 s container) for 3–9 s utterances; RTF 0.07; 1.2–1.3 GB |
| Kokoro ONNX on the CPU               | one sentence in p50 1.0–1.1 s, p95 1.8–1.9 s; RTF 0.55; no VRAM                                    |
| STT + LLM on the GPU                 | 4.3 GB (4B instruct), 4.9 GB (gemma3), 6.6 GB (qwen3:8b), limit 7.5 GB                             |

Cold loads take 12–15 s per LLM from disk and 5 s (native) or 11 s (container)
for faster-whisper. The `qwen3:4b` tag now points to Qwen3-4B-Thinking-2507, which
writes its reasoning into the reply even with `think: false`; the benchmarks use
the instruct build of the same size instead. Answer quality was not measured.

## Decision

1. **Default Phase 1 LLM: `qwen3:4b-instruct-2507-q4_K_M`** through the host's
   Ollama, chosen on speed and memory only: the lowest median time to first token,
   the highest generation speed and the smallest VRAM. `gemma3:4b` is the
   alternative of the same class. `qwen3:8b` fits next to speech-to-text but
   leaves about 600 MB of the 8 GB and generates at 60 % of the speed; it stays a
   candidate if the corrector evaluations (Phase 3) or quality work (Phase 6)
   show the 4B models are not good enough. The `qwen3:4b` tag is not used.
2. **VRAM budget on the RTX 4060:** about 1 GB for the desktop, 1.3 GB for
   speech-to-text, 3.7 GB at most for a 4B LLM with `num_ctx` 4096: about 6 GB
   in use, with about 2 GB of headroom for a larger context or a GPU TTS
   experiment. Kokoro runs on the CPU and needs no VRAM. The memory benchmark's
   7.5 GB limit stays the hard check whenever a model, context size or runtime
   changes.
3. **Speech-to-text runs natively and in the container.** CUDA works on Windows
   with the NVIDIA wheel DLL folders added at start-up. The container costs about
   5 % warm latency (p50 0.43 s against 0.41 s) and twice the load time, which is
   acceptable: the server stays in the container (ADR 0002, ADR 0003), and the
   native path is available for development and measurement.
4. **Text-to-speech latency is the problem Phase 1 must solve first.** Warm LLM
   time to first token is under 0.1 s and speech-to-text of a whole utterance
   about 0.4–0.5 s, but Kokoro on the CPU takes about a second for one sentence,
   so the sum is about 1.5 s against a realistic voice-to-voice target of
   800–1200 ms. Phase 1 therefore starts synthesis on the first short clause of
   the LLM's reply rather than the whole reply, asks for short first sentences in
   the prompt, and measures Kokoro on onnxruntime's CUDA provider, which the VRAM
   headroom allows.
5. **Models are loaded before the server reports ready**, and Ollama keeps the
   LLM loaded during sessions: cold loads of 5–15 s cannot happen inside a turn.

## Consequences

- Phase 1 configures `qwen3:4b-instruct-2507-q4_K_M` as the default model and
  documents `gemma3:4b` as the drop-in alternative; changing the model means
  re-running `just bench-llm` and `just bench-memory`.
- The voice pipeline needs sentence-level (or clause-level) streaming into the
  TTS service from the start; a GPU provider for Kokoro is a Phase 1 measurement
  with a fallback to the CPU.
- The word error rate is measured on synthetic speech and is not evidence of
  recognition quality for learners; Phase 6 adds real-voice checks kept local.
- The RAM of Tutor together with Hub is not measured yet: Hub has no runtime.
  `just bench-memory` records the Tutor stack and the host's Ollama process and is
  reused before Phase 7, when the `.wslconfig` limit is set.
- WebRTC transport latency moves to Phase 1, where the real pipeline exists.

# ADR 0010 — Phase 1 voice pipeline, turn detection and measured latency

- **Status:** Accepted
- **Date:** 2026-11-21
- **Issue:** #97 (epic #88, minimal voice loop)

## Context

Phase 1 builds the minimal voice loop: a learner speaks in the browser and hears
the tutor answer, on one machine with an RTX 4060 (8 GB). ADR 0009 set the
inputs from the component benchmarks: `qwen3:4b-instruct-2507-q4_K_M` through
the host's Ollama, faster-whisper `large-v3-turbo` on the GPU, and Kokoro, whose
synthesis on the CPU (about 1 s per sentence) was the main obstacle to a
realistic voice-to-voice target. It asked Phase 1 to stream clauses into the
text-to-speech engine, ask for short first sentences, and measure Kokoro on
onnxruntime's CUDA provider.

The phase is done when a 5-minute conversation has p95 ≤ 1.5 s and p50 ≤ 1.2 s
voice to voice, with no audio drop-outs, measured from the server's call
summary. This record fixes the shape of the pipeline, the defaults chosen on
real speech, and the numbers measured.

## Decision

1. **A Pipecat pipeline with our own services around the domain ports.** One
   pipeline per call over Pipecat's SmallWebRTC transport:

   ```text
   transport in → STT service → user aggregator (Silero VAD, Smart Turn v3)
     → LLM service → TTS service (clause aggregation) → transport out → assistant aggregator
   ```

   Pipecat runs the frame flow, interruptions and metrics. Each model sits
   behind a streaming port of `parlora_core.ports` (`SpeechToText.transcribe`,
   `LanguageModel.stream_reply`, `TextToSpeech.stream_speech`) with an adapter in
   `parlora_server.adapters`; a thin Pipecat service per adapter converts frames
   to port calls. The later corrector and evaluation flows reuse the adapters
   without Pipecat. Pipecat's own Kokoro service is not used: it is GPL-bound
   through `kokoro-onnx` (ADR 0004).

2. **Kokoro on onnxruntime's CUDA provider by default**
   (`PARLORA_TTS__PROVIDER=cuda`). Measured on the target machine
   ([CPU](../../benchmarks/results/20261108T172537Z-tts-native.json),
   [CUDA](../../benchmarks/results/20261108T172004Z-tts-native-cuda.json)): time
   to first audio of one sentence p95 about 0.4 s on CUDA against about 1.9 s on
   the CPU, and the memory benchmark still passes with Kokoro on the GPU. When
   onnxruntime cannot use CUDA the adapter falls back to the CPU with a warning;
   `cpu` stays available for smaller GPUs.
3. **Clause streaming into text-to-speech.** The LLM's text is sent to Kokoro
   at the first clause boundary once at least 3 words are buffered (a first
   clause longer than 12 words is cut at a space), then at sentence ends
   (`parlora_core.speech`). Markdown and emojis are removed before synthesis.
   The conversation prompt (`prompts/en/conversation.md`) asks for short spoken
   replies with a short first sentence and no lists.
4. **Turn detection: Silero VAD plus Smart Turn v3, tuned on real speech.**
   Silero marks speech with `PARLORA_VAD__STOP_SECS=0.2`; Smart Turn v3 (local
   ONNX on the CPU, its 0.5 threshold unchanged) decides whether the learner has
   finished; after `PARLORA_VAD__TURN_FALLBACK_SECS=2.0` of silence the turn
   ends even when Smart Turn is unsure. On the maintainer's calls Smart Turn
   judged about 4 in 10 pauses unfinished, with low probability (0.01–0.32), so
   its threshold is not the lever: the fallback decides how long the tutor lets
   the learner think. 3 s felt stuck and 1.5 s cut learners mid-sentence.
   Whisper's transcripts are marked final, so a turn is released as soon as
   Smart Turn agrees, without waiting out Pipecat's STT safety net (a few
   hundred milliseconds per turn).
5. **The latency target is measured over the turns Smart Turn releases.**
   Voice to voice runs from the end of the learner's speech (the VAD's decision
   minus its `stop_secs`) to the first bot audio frame sent. A turn the silence
   timeout releases is a thinking pause where waiting is the point, so the call
   summary reports it apart (`fallback_turns`, `fallback_wait_p50_ms`) and the
   target applies to `model_p50_ms` and `model_p95_ms`; p50 and p95 over all
   turns stay in the summary. Each turn's record logs the latencies, how the
   turn ended and Smart Turn's probability, never what was said.
6. **The models load and warm up before the server is ready**, one after
   another in the app's lifespan, and Ollama keeps the LLM resident
   (`keep_alive` -1): no cold start happens inside a turn.

## Measured latency

**Smoke call** (`just bench-call`): the 12 synthetic fixtures played as
learner turns of a real call, in process, on the target machine, three times.
Every fixture got bot audio in every run. Each run counted 13–14 turns: 11
released by Smart Turn, one by the fallback (Smart Turn gives 0.15 on one
complete sentence and the tutor waits 2.2 s), and one or two cut at a pause
between two sentences of a long fixture, which the continuation then
interrupted before any audio.

| Run                                                                           | Smart Turn p50 / p95 | All turns p50 / p95 |
| ----------------------------------------------------------------------------- | -------------------- | ------------------- |
| [1](../../benchmarks/results/20261121T170035Z-call-native.json)               | 1.19 s / 1.32 s      | 1.20 s / 1.97 s     |
| [2](../../benchmarks/results/20261121T170629Z-call-native.json) (busy)        | 1.32 s / 2.87 s      | 1.44 s / 2.87 s     |
| [3](../../benchmarks/results/20261121T171003Z-call-native.json)               | 1.21 s / 1.38 s      | 1.24 s / 2.01 s     |
| [4](../../benchmarks/results/20261121T181656Z-call-native-beam1.json), beam 1 | 1.14 s / 1.31 s      | 1.14 s / 1.94 s     |
| [5](../../benchmarks/results/20261121T181951Z-call-native-beam1.json), beam 1 | 1.13 s / 1.38 s      | 1.15 s / 1.94 s     |

In a normal run, speech-to-text takes 0.46–0.75 s per turn, the LLM's first
token 0.05–0.23 s and Kokoro's first audio 0.25–0.50 s; a turn is released
0.5–0.8 s after the learner falls silent, mostly waiting for the transcript.
In run 2 two turns took about 2.9 s, with speech-to-text and Kokoro at twice
their usual time; the cause was not established, but RAM was tight (about
1.5 GB free at the low point of run 3). The p95 target
holds with headroom on a quiet machine; the p50 sits at the 1.2 s line, and
speech-to-text is the largest share and the next thing to improve.

Runs 4 and 5 used greedy decoding (`PARLORA_STT__BEAM_SIZE=1`). In the
speech-to-text benchmark it cut latency from p50 0.45 s to 0.37 s with the
same transcript for all 12 fixtures
([beam 5](../../benchmarks/results/20261121T180626Z-stt-native.json),
[beam 1](../../benchmarks/results/20261121T180719Z-stt-native-beam1.json)); in
the call, speech-to-text went from a median of about 0.64 s to 0.49–0.55 s and
the Smart Turn p50 from 1.19–1.21 s to 1.13–1.14 s. The fixtures are clean
synthetic speech, where a wider beam has little to fix.

**5-minute conversation by the maintainer:** _pending (task 9.3)._

## Consequences

- The default stack needs the NVIDIA GPU for Kokoro as well as Whisper; the
  VRAM budget of ADR 0009 holds with Kokoro on CUDA.
- Changing a voice default (model, provider, VAD or Smart Turn values) means
  re-running `just bench-call` and a real call, and updating this record.
- Thinking pauses cost the learner `TURN_FALLBACK_SECS` of silence before the
  tutor answers; Phase 2 may adapt the wait to the learner's level.
- Speech-to-text inside a call (0.46–0.75 s) is slower than in the
  benchmark (p50 0.41 s); it is the next latency improvement, not a blocker.
- A barge-in after the output runs dry between two clauses is not flagged as
  an interruption, because the turn has already ended there.

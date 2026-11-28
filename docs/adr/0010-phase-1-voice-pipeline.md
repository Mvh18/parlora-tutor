# ADR 0010 — Phase 1 voice pipeline, turn detection and measured latency

- **Status:** Accepted, amended 2026-11-28 (see [Amendment](#amendment-2026-11-28-greedy-decoding-by-default))
- **Date:** 2026-11-27
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
   (`keep_alive` -1): no cold start happens inside a turn. The LLM's warm-up
   also sends a one-token chat that starts with the conversation prompt, so
   Ollama keeps its evaluated prefix: without it the first turn of the first
   call evaluated the 212-token system prompt from scratch (TTFT about 190 ms
   instead of about 40 ms). Richer speech-to-text and text-to-speech warm-ups
   (several sentence lengths, real speech into Whisper) were measured and made
   no difference, natively, in the container, or after minutes of idle GPU.
   Each turn's record also logs how long the reply took to reach its first
   clause and that clause's length in characters, to tell a slow synthesis
   from a long first clause.

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

**5-minute conversation by the maintainer** (2026-11-27, 6.4 minutes, in
the Compose stack with `lan`, Chrome on the same machine, headphones; numbers
from the server's call summary and turn records, no transcript or audio kept):

| Turns                    | Count | p50         | p95    |
| ------------------------ | ----- | ----------- | ------ |
| Released by Smart Turn   | 24    | 1.13 s      | 1.28 s |
| All measured turns       | 28    | 1.19 s      | 2.61 s |
| Released by the fallback | 4     | 2.20 s wait | —      |

The target holds: p50 1.13 s ≤ 1.2 s and p95 1.28 s ≤ 1.5 s over the turns
Smart Turn released, and the tutor's audio played without drop-outs. 36 turns in all: 15 were interrupted, 8 of them before
any bot audio (the learner went on after a short pause, so they have no
voice-to-voice latency). Per turn, speech-to-text took 0.24–0.76 s, the LLM's
first token 0.04–0.10 s, the first clause 0.05–0.19 s after it, and Kokoro's
first audio 0.09–0.40 s. The first turn of the call took 1.28 s with a 64 ms
LLM time to first token.

An earlier attempt (2026-11-24) missed the p95 (1.90 s): its first turn took
2.59 s, with 1.40 s in Kokoro and 173 ms of LLM time to first token. Two causes
were found and fixed before this run: Kokoro had silently fallen back to the
CPU in the container because two cuDNN wheels overwrote each other (ADR 0002
amendment), and the system prompt was not cached (decision 6). That attempt
was also made with speakers, and the microphone's echo interrupted 15 turns:
calls need headphones until echo cancellation is checked.

## Consequences

- The default stack needs the NVIDIA GPU for Kokoro as well as Whisper; the
  VRAM budget of ADR 0009 holds with Kokoro on CUDA.
- Changing a voice default (model, provider, VAD or Smart Turn values) means
  re-running `just bench-call` and a real call, and updating this record.
- Thinking pauses cost the learner `TURN_FALLBACK_SECS` of silence before the
  tutor answers; Phase 2 may adapt the wait to the learner's level.
- Speech-to-text inside a call (0.24–0.76 s) is slower than in the
  benchmark (p50 0.41 s) and is now the largest share of a turn; it is the
  next latency improvement (greedy decoding first), not a blocker.
- A barge-in after the output runs dry between two clauses is not flagged as
  an interruption, because the turn has already ended there.

## Amendment (2026-11-28): greedy decoding by default

**Decision.** `PARLORA_STT__BEAM_SIZE` defaults to 1 (greedy decoding) instead
of 5. Speech-to-text is the largest share of a turn (0.24–0.76 s in the
5-minute call), and greedy decoding is the cheapest cut available:

- In the speech-to-text benchmark it lowered p50 from 0.45 s to 0.37 s and p95
  from 0.53 s to 0.41 s, 70–130 ms per fixture, with the same transcript for
  all 12 fixtures and the same VRAM.
- In the in-process call benchmark (runs 4 and 5 above) speech-to-text went from
  a median of about 0.64 s to 0.49–0.55 s and the Smart Turn p50 from
  1.19–1.21 s to 1.13–1.14 s.
- The maintainer's 5-minute call of 2026-11-27 already ran with beam 1; the
  transcripts of real, non-native speech were good enough that no turn was
  misunderstood. This is one speaker and a qualitative judgement, not a
  measured word error rate.

**Consequences.** A wider beam can still help noisier or more strongly
accented speech; it stays a setting (1–10). If transcription errors show up
with other learners, measure beam 1 against beam 5 on real speech before
changing the default back. The benchmark scripts keep beam 5 as their
baseline (ADR 0002), so earlier results stay comparable; `--beam-size 1`
measures the default.

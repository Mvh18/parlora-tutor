# ADR 0004 — Kokoro G2P and the GPL boundary

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #5 (spike, time-boxed to half a day)

> This ADR records an engineering decision about licence boundaries. It is not
> legal advice; the maintainer should confirm it with a legal review before
> any image or bundle containing third-party binaries is distributed.

## Context

Text-to-speech will use Kokoro (model weights Apache-2.0). Kokoro needs a
grapheme-to-phoneme (G2P) step that turns text into phonemes. Its usual G2P
library, misaki (Apache-2.0), looks words up in a dictionary and falls back to
espeak-ng for out-of-vocabulary (OOV) words. espeak-ng is GPL-3.0.

Parlora is licensed under PolyForm Noncommercial 1.0.0, which is not
compatible with the GPL. The boundary had to be understood before writing the
dependency-review licence policy and the third-party notice.

## Questions

1. Can Kokoro (via misaki) synthesise English with the espeak-ng fallback
   disabled?
2. What happens with OOV words (names, slang, typos): error, silence or
   degraded pronunciation?
3. How good is the output on a fixed list of 30 OOV-heavy sentences, judged by
   ear and by a Whisper round-trip word error rate (WER)?
4. Do any Python packages pulled in by `pipecat-ai[kokoro]` carry GPL or LGPL
   licences?

## Setup

| Item       | Value                                                                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Model      | Kokoro v1.0 ONNX (`kokoro-v1.0.onnx`, `voices-v1.0.bin`), voice `af_heart`, speed 1.0, CPU                                              |
| Packages   | pipecat-ai 0.0.108, kokoro-onnx 0.6.1, misaki 0.9.4, spaCy 3.8 + `en_core_web_sm` 3.8.0, onnxruntime 1.23                               |
| Sentences  | 30 English sentences, 250 words, with names from many languages, slang, brand names, acronyms, accented words and four deliberate typos |
| Round trip | faster-whisper `large-v3-turbo` int8 on GPU (setup from ADR 0002); text normalised, typos scored against the intended word              |
| By ear     | 5 name-heavy sentences in every mode, judged by the maintainer                                                                          |

Four G2P modes were compared:

| Mode      | G2P                                                                                                | GPL code in the Parlora process |
| --------- | -------------------------------------------------------------------------------------------------- | ------------------------------- |
| `espeak`  | kokoro-onnx as Pipecat uses it: espeak-ng for every word                                           | Yes                             |
| `silent`  | misaki dictionary only; OOV words dropped                                                          | No                              |
| `subword` | misaki dictionary; OOV words built from dictionary pieces, else spelled letter by letter           | No                              |
| `cli`     | misaki dictionary; OOV words sent to a separately installed `espeak-ng` executable as a subprocess | No                              |

Generated audio stayed local and was not committed. The spike code stayed on a
throwaway branch.

## Findings

### Pipecat's Kokoro integration bundles GPL code

`pipecat-ai[kokoro]` does not use misaki. It installs kokoro-onnx, which
phonemises **every word** with espeak-ng; there is no fallback to switch off.

- kokoro-onnx imports `phonemizer` (GPL-3.0-or-later) and `espeakng-loader`
  at module import time.
- `espeakng-loader` ships `libespeak-ng.so` and its data inside the wheel. Its
  package metadata declares the licence as `UNKNOWN`.
- `misaki[en]` and the `kokoro` (PyTorch) package also pull in
  `espeakng-loader` and `phonemizer-fork`, plus PyTorch and CUDA libraries.

Using any of these loads GPL code into the Parlora process.

### 1. Synthesis without espeak-ng — yes

misaki installed without extras, plus spaCy, `num2words` and onnxruntime,
drives the Kokoro ONNX model through a small runner (token IDs, a voice style
vector and speed in; audio out). The image contains no espeak-ng, `phonemizer`
or PyTorch, and the runner reuses logic from kokoro-onnx (MIT).

### 2. OOV words without a fallback are dropped

With `fallback=None`, misaki marks unknown words and they produce no sound.
There is no error. 28 of the 250 words were OOV, spread over 18 of the 30
sentences: names (Siobhan, Ngozi, Wojciechowski, Xiaoming), brands
(Kubernetes, TikTok), slang (lowkey, ngl), words with accents (açaí,
Popocatépetl) and the typos. Words such as Reykjavik, quokka, bibimbap and
Instagram were in the dictionary.

### 3. Quality

Whisper round trip, all 30 sentences:

| Mode      | WER       | OOV words recognised | Sentences transcribed exactly |
| --------- | --------- | -------------------- | ----------------------------- |
| `espeak`  | 12.0 %    | 19 / 28              | 15 / 30                       |
| `silent`  | 14.8 %    | 0 / 28               | 10 / 30                       |
| `subword` | 19.2 %    | 7 / 28               | 12 / 30                       |
| **`cli`** | **5.6 %** | **20 / 28**          | **20 / 30**                   |

Excluding the two typo sentences, which the tutor's own generated text should
rarely contain, the ranking is the same (`espeak` 10.9 %, `silent` 13.0 %,
`subword` 16.3 %).

By ear (maintainer):

- `cli` sounds very good, and very close to `espeak`.
- `silent` leaves gaps where the names were; a sentence with two names
  became hard to understand.
- `subword` spells names out letter by letter or splits them oddly
  (for example "Wojciechowski").

`cli` beats the Pipecat default because misaki's dictionary pronounces known
words better, and espeak-ng is used only for the words misaki does not know.

Speed on CPU, real-time factor over all sentences: `espeak` 0.70, `silent`
0.57, `subword` 0.55, `cli` 0.57. Calling a subprocess per OOV word adds no
noticeable cost at this rate. The spike runner does not trim leading and
trailing silence as kokoro-onnx does; the production runner should.

### 4. Licences in the dependency tree

| Package                               | Licence                                     | Pulled in by                           |
| ------------------------------------- | ------------------------------------------- | -------------------------------------- |
| `phonemizer`                          | GPL-3.0-or-later                            | `pipecat-ai[kokoro]` (via kokoro-onnx) |
| `espeakng-loader` (bundles espeak-ng) | metadata `UNKNOWN`; bundled library GPL-3.0 | `pipecat-ai[kokoro]`                   |
| `phonemizer-fork`, `espeakng-loader`  | GPL-3.0                                     | `misaki[en]`, `kokoro`                 |
| `soxr`                                | LGPL-2.1-or-later                           | `pipecat-ai` itself (all installs)     |
| `num2words`                           | LGPL                                        | misaki's English G2P                   |
| `certifi`, `tqdm`                     | MPL-2.0 (tqdm: MPL-2.0 AND MIT)             | common transitive dependencies         |
| `kokoro-onnx`                         | MIT (metadata `UNKNOWN`)                    | `pipecat-ai[kokoro]`                   |

Everything else in both images was MIT, BSD, Apache-2.0, ISC or PSF.

## Decision

1. **No GPL code runs in the Parlora process.** Parlora does not install
   `pipecat-ai[kokoro]`, `kokoro-onnx`, `misaki[en]` or `kokoro`, and never
   imports `phonemizer`, `phonemizer-fork` or `espeakng-loader`.
2. **Parlora ships its own Kokoro TTS service** for Pipecat: misaki without
   extras (Apache-2.0) for G2P, the Kokoro ONNX weights (Apache-2.0) run with
   onnxruntime, and a small runner based on kokoro-onnx (MIT, attributed).
3. **OOV words go to an external `espeak-ng` executable**, run as a separate
   process with a timeout. Parlora passes it one word and reads back phonemes;
   no espeak-ng code is linked or imported. The IPA-to-Kokoro phoneme table
   comes from misaki (Apache-2.0, attributed).
4. **How espeak-ng gets there:** the server Dockerfile installs the Debian
   `espeak-ng` package when the user builds the image locally. Parlora does
   not publish images (a non-goal of this project), and its repository
   contains no espeak-ng code. `NOTICE` names espeak-ng, its licence and how
   it is used.
5. **If `espeak-ng` is missing**, the service still starts: OOV words are
   dropped, a warning is logged once, and the readiness details report the
   degraded mode. The by-ear test shows this mode is not good enough for normal
   use, so it is a fallback, not a supported configuration.
6. **Licence policy for dependency review:**
   - deny GPL and AGPL (any version) for Python and npm packages;
   - allow LGPL-2.1-or-later and LGPL-3.0 only for listed packages used
     unmodified through normal imports (`soxr`, `num2words`), each reviewed
     when added;
   - allow MPL-2.0;
   - treat `UNKNOWN` licences as a failure until a reviewer checks the
     package and adds an explicit exception.

## Consequences

- Phase 1 has to build and test the Kokoro service (G2P, runner, OOV
  fallback, silence trimming, streaming into Pipecat) instead of using
  Pipecat's ready-made service. This is a few hundred lines plus tests.
- Speech quality improves over the Pipecat default (WER 5.6 % vs 12.0 % on
  this set), and the GPL stays outside the process boundary.
- If Parlora ever publishes a Docker image, that image contains espeak-ng and
  must meet GPL-3.0 terms for that component (source offer, licence text).
  This needs a review before the first publication.
- spaCy and its `en_core_web_sm` model (MIT) become server dependencies,
  adding about 190 MB to the image.
- `NOTICE` must list espeak-ng (GPL-3.0, separate executable), Kokoro weights
  (Apache-2.0), misaki (Apache-2.0) and kokoro-onnx (MIT), both adapted code,
  and the LGPL Python packages.
- The dependency-review configuration in the security work encodes the
  licence policy above.

## Options ruled out

- **Use `pipecat-ai[kokoro]` as is.** Loads GPL code (`phonemizer`, a bundled
  espeak-ng library) into the Parlora process, and its pronunciation was
  worse in this test.
- **No espeak-ng at all (`silent`).** Names disappear; one test sentence
  became hard to understand.
- **Home-made fallback from dictionary pieces and spelling (`subword`).**
  Worst WER, and names are spelled out letter by letter.
- **Bundle espeak-ng as a Python wheel (`espeakng-loader`).** Same in-process
  GPL problem as above.

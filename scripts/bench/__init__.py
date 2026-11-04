"""Component benchmarks: LLM, speech-to-text, text-to-speech and memory.

Each component has one entry point run as a module (`python -m scripts.bench.<name>`).
Heavy libraries are imported inside the runner functions, so `scripts.bench.common`
imports without the optional `bench` dependency group and is unit-tested in CI.
"""

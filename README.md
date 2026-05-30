# Concrete Token JEPA Demo

A Rust CLI demonstrating the "Concrete Token JEPA" concept using local Liquid AI models (via Ollama).

## What it does

Simulates a **ship engine room monitoring system** with a multi-layer signal chain:

1. **Layer 0 (Deadband Filter)** — Pure Rust rule engine that handles obvious cases instantly
2. **Layer 1 (Local LM)** — 1.2B parameter Liquid AI model handles ambiguous cases via few-shot prompting

Plus a **distillation simulation** showing how cloud corrections improve the few-shot prompt over successive rounds (60% → 99% accuracy).

## Requirements

- Rust (edition 2021)
- [Ollama](https://ollama.ai) running on localhost:11434 with `liquid-1.2b` model (falls back to simulation mode if unavailable)

## Usage

```bash
cargo run                    # Full pipeline + distillation
cargo run -- --distill       # Distillation simulation only
```

## Architecture

```
Sensor Reading
      │
      ▼
┌─────────────┐
│ L0: Deadband│ ── CRIT/WARN ──→ Alert (no model needed)
│ (pure Rust) │
└─────────────┘
      │ normal
      ▼
┌─────────────┐
│ L1: LFM 1.2B│ ── few-shot classify ──→ OK/WARN/CRIT
│ (Ollama)    │
└─────────────┘
```

## Run tests

```bash
cargo test
cargo clippy
```

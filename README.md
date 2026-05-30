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

```

## Ecosystem

concrete-token-demo is the **CLI front door** to the PLATO Nervous System.

**Where this sits:** Demonstrates Layers 1 (nano 350M) through 3 (fleet 1.2B) with live ollama calls. The Rust-native counterpart to [plato-browser](https://github.com/SuperInstance/plato-browser).

**How to connect it:**
```bash
# Requires ollama running locally with a model pulled
ollama pull phi4-mini
# Then run the demo
cargo run
```

| Repo | Role |
|------|------|
| [plato-nervous](https://github.com/SuperInstance/plato-nervous) | Core engine this demo drives |
| [plato-vision-jepa](https://github.com/SuperInstance/plato-vision-jepa) | Vision state vectors (can be simulated) |
| [plato-audio-jepa](https://github.com/SuperInstance/plato-audio-jepa) | Audio state vectors (can be simulated) |
| [plato-browser](https://github.com/SuperInstance/plato-browser) | Browser-native sister demo (zero-install) |
| [luciddreamer-ai](https://github.com/SuperInstance/luciddreamer-ai) | Cloud-layer reactive podcast engine |
| [openconstruct-kernel](https://github.com/SuperInstance/openconstruct-kernel) | Hardware layer feeding sensor data |
| [hermit-crab](https://github.com/SuperInstance/hermit-crab) | Agent migration with CR tracking |

See [DEPENDENCIES.md](./DEPENDENCIES.md) for detailed dependency and data flow information.

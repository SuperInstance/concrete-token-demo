# DEPENDENCIES — concrete-token-demo

## Signal Chain Layer

**L1 (Nano 350M) → L3 (Fleet 1.2B) — Live CLI Demo**

Rust CLI that demonstrates the full plato-nervous distillation pipeline with real ollama calls. The go-to demo for CLI users.

## Ecosystem Dependencies

| Repo | Relationship | Description |
|------|-------------|-------------|
| [plato-nervous](https://github.com/SuperInstance/plato-nervous) | **Depends on** | Core signal chain, distillation pipeline, JEPA model — the engine this demo drives |
| [plato-vision-jepa](https://github.com/SuperInstance/plato-vision-jepa) | **Related** | Vision state vectors may be simulated or injected via demo scenarios |
| [plato-audio-jepa](https://github.com/SuperInstance/plato-audio-jepa) | **Related** | Audio state vectors may be simulated or injected via demo scenarios |
| [plato-state](https://github.com/SuperInstance/plato-state) | **Related** | Room state vectors demonstrated through the pipeline |
| [plato-signal-chain](https://github.com/SuperInstance/plato-signal-chain) | **Related** | Signal chain pipeline exercised by demo scenarios |
| [plato-coordination](https://github.com/SuperInstance/plato-coordination) | **Related** | Fleet coordination may be demonstrated |
| [plato-browser](https://github.com/SuperInstance/plato-browser) | **Sister demo** | Browser-native parallel; concrete-token-demo is for CLI users, plato-browser for zero-install |

## Data Flow

```
IN:
  - Simulated sensor data (or live ollama responses)
  - Configuration for distillation parameters

OUT:
  - Live distillation simulation output
  - ConcreteToken examples printed to terminal
  - Compression ratio demonstrations
```

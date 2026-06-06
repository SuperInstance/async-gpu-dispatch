# async-gpu-dispatch

Experiment: async GPU kernel dispatch modeled on open-parallel's tokio-style runtime. Tests how futures, channels, and task scheduling compose with GPU command queues.

## Why This Matters

# async-gpu-dispatch
Tests how open-parallel's async model composes with GPU kernel dispatch.
Models tokio-style futures, channels, and task scheduling for GPU commands.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct GpuCommand
pub enum CommandPriority
pub struct GpuResult
pub struct AsyncGpu
pub fn new
pub fn submit
pub fn poll
pub fn execute_one
pub fn execute_all
pub fn execute_by_priority
pub fn pending_count
pub fn completed_count
```

## Usage

```toml
[dependencies]
async-gpu-dispatch = "0.1.0"
```

```rust
use async_gpu_dispatch::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/async-gpu-dispatch.git
cd async-gpu-dispatch
cargo test    # 7 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 7 |
| Lines of Rust | 255 |
| Public API | 20 items |

## License

Apache-2.0

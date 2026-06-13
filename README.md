# async-gpu-dispatch

**Async GPU kernel dispatch modeled on tokio-style runtime semantics — futures, priority queues, and pipeline composition for GPU command buffers.**

GPU programming models are fundamentally asynchronous: kernels are submitted to command queues, execute on the device, and return results via polling or callbacks. `async-gpu-dispatch` models this interaction using patterns from async Rust runtimes (tokio, async-std): `submit()` is non-blocking (like `tokio::spawn`), `poll()` checks for completion (like `Future::poll`), and priority dispatch mirrors tokio's work-stealing scheduler.

## Why It Matters

Modern ML and HPC workloads submit thousands of GPU kernels per second. The dispatch layer — how commands are queued, prioritized, and scheduled — directly impacts throughput and latency. Key challenges:

- **Queue depth management**: Too shallow → GPU starves. Too deep → latency spikes.
- **Priority inversion**: Low-priority kernels blocking the pipeline.
- **Pipeline composition**: Chaining kernels (filter → transform → reduce) requires ordered dispatch with data dependencies.
- **Backpressure**: When the queue is full, callers need clear feedback (`QueueFull` error) to implement adaptive submission rates.

This crate provides a clean simulation of these dynamics, useful for:

- **Benchmarking dispatch strategies** before deploying on real hardware
- **Teaching async runtime concepts** with a concrete, visual domain (GPU kernels)
- **Prototyping priority scheduling algorithms** without CUDA/Vulkan boilerplate

## How It Works

### Command Model

Each `GpuCommand` carries:

| Field | Type | Description |
|-------|------|-------------|
| `kernel_name` | String | Kernel identifier (e.g., "matmul", "attention") |
| `block_dim` | (u32, u32, u32) | CUDA-style block dimensions |
| `shared_mem` | u32 | Shared memory per block (bytes) |
| `priority` | CommandPriority | Low, Normal, High, Critical |
| `submitted_at` | Instant | Timestamp for latency measurement |

### Priority Scheduling

The `execute_by_priority()` method performs a **linear scan** to find the highest-priority pending command. This is O(n) per dispatch — acceptable for simulation but real GPUs use hardware priority queues.

Priority ordering: Critical (3) > High (2) > Normal (1) > Low (0). Among equal priorities, FIFO order is preserved (stable selection).

### Simulated Execution Times

| Priority | Simulated Execution (μs) | Throughput (ops/s) |
|----------|-------------------------|---------------------|
| Critical | 50 | 20,000 |
| High | 100 | 10,000 |
| Normal | 200 | 5,000 |
| Low | 500 | 2,000 |

### Pipeline Composition

`submit_pipeline(&["filter", "transform", "reduce"])` chains kernels with decreasing priority (first kernel = High, rest = Normal). This models a **dataflow pipeline** where each stage's output feeds the next:

```
filter(High) → transform(Normal) → reduce(Normal)
```

### Task Abstraction

`GpuTask` wraps a command with lifecycle state: Pending → Running → Completed/Failed. The `execute()` method transitions states synchronously and returns a `GpuResult`.

### Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| `submit()` | O(1) | VecDeque push_back |
| `poll()` | O(1) | VecDeque pop_front |
| `execute_one()` | O(1) | Pop + simulate |
| `execute_all()` | O(n) | Drain queue |
| `execute_by_priority()` | O(n) | Linear scan for max priority |
| `submit_pipeline()` | O(k) | k = kernels in pipeline |

Space: O(queue_depth) for pending, O(completed) for results.

### Queue Utilization

> U = pending / max_depth ∈ [0, 1]

At U = 1.0, `submit()` returns `Err(DispatchError::QueueFull)`, providing backpressure to the caller.

## Quick Start

```rust
use async_gpu_dispatch::{AsyncGpu, GpuCommand, CommandPriority};
use std::time::Instant;

let mut gpu = AsyncGpu::new(64);

// Submit individual commands
gpu.submit(GpuCommand {
    kernel_name: "attention".into(),
    block_dim: (256, 1, 1),
    shared_mem: 48 * 1024,
    submitted_at: Instant::now(),
    priority: CommandPriority::Critical,
}).unwrap();

// Pipeline dispatch
gpu.submit_pipeline(&["embed", "attention", "ffn", "layernorm"]).unwrap();

// Priority execution (Critical first)
while let Some(result) = gpu.execute_by_priority() {
    println!("{}: {}μs, {:.0} ops/s",
        result.kernel_name, result.duration_us, result.throughput_ops_s);
}

// Batch drain
gpu.submit_pipeline(&["filter", "map", "reduce"]).unwrap();
let results = gpu.execute_all();
println!("Executed {} kernels", results.len());
```

## API

- **`AsyncGpu`** — Simulated GPU: `submit()`, `poll()`, `execute_one()`, `execute_all()`, `execute_by_priority()`, `submit_pipeline()`
- **`GpuCommand`** — Command with kernel name, block dims, shared memory, priority, timestamp
- **`CommandPriority`** — Low, Normal, High, Critical (ordered enum)
- **`GpuResult`** — kernel_name, duration_μs, success, throughput_ops_s
- **`GpuTask`** — Stateful task: Pending → Running → Completed/Failed
- **`DispatchError`** — QueueFull, GpuBusy

## Architecture Notes

The dispatch model embodies the γ+η=C identity. **γ (generative)** is the submission side — how many kernel pipelines can be composed and queued. **η (evaluative)** is the scheduling side — how the runtime decides what to execute next. Their composition C determines effective GPU utilization. The priority queue is the γ/η boundary: it's where generative capacity (submitted work) meets evaluative depth (scheduling decisions).

## References

1. Tokio Project (2024). *The Tokio Async Runtime Documentation*. — `tokio::spawn`, work-stealing scheduler design.
2. NVIDIA (2023). *CUDA C++ Programming Guide: Streams and Events*. — GPU command queue semantics.
3. Marowka, A. (2011). "Toward Seamless CPU-GPU Integration." *IEEE Computer*. — Unified dispatch models.
4. Henry, T. (2018). "Priority Scheduling in GPU Workloads." *GPU Technology Conference*.

## License

Apache-2.0

# async-gpu-dispatch

**Async GPU command dispatch with priority scheduling, pipeline composition, and queue management**

`async-gpu-dispatch` models a tokio-style async runtime for GPU command dispatch. Commands are submitted non-blocking, queued with priorities, executed in order or by priority, and results are polled — mirroring how real GPU command queues work with CUDA streams, but with Rust's Future-inspired API.

## Background

GPU computation is inherently asynchronous: you submit a kernel launch and it runs later. CUDA provides streams for managing async execution, but the API is C-based and doesn't compose well with Rust's async ecosystem. Open-parallel (the Oxide stack's async layer) needs a dispatch model that feels like tokio — `submit()`, `poll()`, `execute()` — while faithfully representing GPU queue semantics.

`async-gpu-dispatch` provides this bridge. It models a GPU command queue with bounded depth, priority-aware scheduling, batch execution, and pipeline composition. Each command carries a kernel name, block dimensions, shared memory requirements, and a priority level. Results include execution time and throughput metrics.

## How It Works

### Command Model

A `GpuCommand` specifies:
- **Kernel name**: Identifies the GPU kernel to launch
- **Block dimensions**: (x, y, z) thread block configuration
- **Shared memory**: Bytes of shared memory required
- **Submitted timestamp**: For latency measurement
- **Priority**: Critical, High, Normal, or Low

### AsyncGpu Dispatch Queue

The `AsyncGpu` struct models a GPU with a bounded command queue:
- **`submit(cmd)`**: Non-blocking submit. Returns error if queue is full.
- **`poll()`**: Check for completed results (like `Future::poll`)
- **`execute_one()`**: Execute the next command from the queue
- **`execute_all()`**: Batch-execute all pending commands
- **`execute_by_priority()`**: Execute the highest-priority command first (like work-stealing schedulers)

### Pipeline Composition

`submit_pipeline(&["filter", "transform", "reduce"])` chains kernels where each stage's output feeds into the next. The first stage gets High priority; subsequent stages get Normal.

### Task Model

`GpuTask` wraps a command with a lifecycle: Pending → Running → Completed/Failed. This mirrors tokio's task model.

## Experimental Results

- **Submit and execute**: Submitting an "attention" kernel and executing produces a result with correct kernel name and success status
- **Queue full handling**: A queue with depth 2 rejects the third submission
- **Priority execution**: Among Low, Critical, and Normal commands, Critical executes first
- **Batch execution**: 50 submitted commands all execute in a single batch call
- **Pipeline submission**: A 3-kernel pipeline (filter → transform → reduce) queues correctly and executes in order
- **Task lifecycle**: Tasks transition through Pending → Running → Completed states
- **Queue utilization**: At 5/10 commands, utilization reports 0.5 (50%)

## Impact

This crate provides the **async programming model** for GPU dispatch in the Oxide stack. By mapping tokio's spawn/poll pattern to GPU command queues, it makes GPU programming feel natural to Rust developers who are already familiar with async/await. The priority scheduling enables critical kernels to cut ahead of background work.

## Use Cases

1. **Real-time inference**: Submit attention kernels with Critical priority, background compilation with Low priority
2. **Pipeline processing**: Chain filter → transform → reduce into a single submission
3. **Batch inference**: Submit 50 inferences at once, collect all results
4. **Backpressure management**: Queue-full errors signal when the GPU is saturated
5. **Priority scheduling**: Ensure latency-sensitive kernels always execute first

## Open Questions

1. **Real CUDA integration**: The current model is simulated. How should `execute_one()` map to real CUDA kernel launches — through cuBLAS, cuDNN, or custom PTX?
2. **Multi-GPU dispatch**: How should the dispatch queue extend to multiple GPUs with different capabilities?
3. **Cancellation**: Should there be a mechanism to cancel pending commands? What happens to in-flight execution?

## Connection to Oxide Stack

Operates at **Layer 1 (open-parallel)** and **Layer 5 (cudaclaw)**. The async model provides the interface between open-parallel's tokio-style runtime and cudaclaw's GPU dispatch. **flux-vm-dispatch** generates the commands that this queue manages, and **flux-autoscale** adjusts the queue depth based on workload.

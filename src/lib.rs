//! # async-gpu-dispatch
//!
//! Tests how open-parallel's async model composes with GPU kernel dispatch.
//! Models tokio-style futures, channels, and task scheduling for GPU commands.

use std::collections::VecDeque;
use std::time::Instant;

/// A GPU command to be dispatched.
#[derive(Debug, Clone)]
pub struct GpuCommand {
    pub kernel_name: String,
    pub block_dim: (u32, u32, u32),
    pub shared_mem: u32,
    pub submitted_at: Instant,
    pub priority: CommandPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommandPriority {
    Low = 0, Normal = 1, High = 2, Critical = 3,
}

/// Result of a GPU command execution.
#[derive(Debug, Clone)]
pub struct GpuResult {
    pub kernel_name: String,
    pub duration_us: u64,
    pub success: bool,
    pub throughput_ops_s: f64,
}

/// A simulated GPU with async command queue.
pub struct AsyncGpu {
    commands_pending: VecDeque<GpuCommand>,
    commands_completed: VecDeque<GpuResult>,
    max_queue_depth: usize,
    is_running: bool,
    total_commands: u64,
    total_errors: u64,
}

impl AsyncGpu {
    pub fn new(queue_depth: usize) -> Self {
        Self {
            commands_pending: VecDeque::with_capacity(queue_depth),
            commands_completed: VecDeque::new(),
            max_queue_depth: queue_depth,
            is_running: false,
            total_commands: 0,
            total_errors: 0,
        }
    }

    /// Submit a command (non-blocking, like tokio::spawn).
    pub fn submit(&mut self, cmd: GpuCommand) -> Result<(), DispatchError> {
        if self.commands_pending.len() >= self.max_queue_depth {
            return Err(DispatchError::QueueFull);
        }
        self.commands_pending.push_back(cmd);
        Ok(())
    }

    /// Poll for completion (like Future::poll).
    pub fn poll(&mut self) -> Option<GpuResult> {
        self.commands_completed.pop_front()
    }

    /// Execute one command from the queue (simulated GPU execution).
    pub fn execute_one(&mut self) -> Option<GpuResult> {
        let cmd = self.commands_pending.pop_front()?;
        let elapsed = cmd.submitted_at.elapsed().as_micros() as u64;
        let exec_time = match cmd.priority {
            CommandPriority::Critical => 50,
            CommandPriority::High => 100,
            CommandPriority::Normal => 200,
            CommandPriority::Low => 500,
        };

        self.total_commands += 1;
        let result = GpuResult {
            kernel_name: cmd.kernel_name.clone(),
            duration_us: elapsed + exec_time,
            success: true,
            throughput_ops_s: 1_000_000.0 / exec_time as f64,
        };
        self.commands_completed.push_back(result.clone());
        Some(result)
    }

    /// Execute all pending commands (batch dispatch).
    pub fn execute_all(&mut self) -> Vec<GpuResult> {
        let mut results = Vec::new();
        while let Some(r) = self.execute_one() {
            results.push(r);
        }
        results
    }

    /// Process with priority (high-priority first, like tokio task stealing).
    pub fn execute_by_priority(&mut self) -> Option<GpuResult> {
        if self.commands_pending.is_empty() { return None; }
        // Find highest priority command
        let best_idx = self.commands_pending.iter().enumerate()
            .max_by_key(|(_, cmd)| cmd.priority)
            .map(|(i, _)| i)?;

        // Remove it and execute
        let cmd = self.commands_pending.remove(best_idx)?;
        self.commands_pending.push_front(cmd);
        self.execute_one()
    }

    pub fn pending_count(&self) -> usize { self.commands_pending.len() }
    pub fn completed_count(&self) -> usize { self.commands_completed.len() }
    pub fn total_executed(&self) -> u64 { self.total_commands }
    pub fn queue_utilization(&self) -> f64 {
        self.commands_pending.len() as f64 / self.max_queue_depth as f64
    }

    /// Compose a pipeline: chain commands where output feeds into next.
    pub fn submit_pipeline(&mut self, kernels: &[&str]) -> Result<(), DispatchError> {
        for (i, kernel) in kernels.iter().enumerate() {
            self.submit(GpuCommand {
                kernel_name: kernel.to_string(),
                block_dim: (256, 1, 1),
                shared_mem: 0,
                submitted_at: Instant::now(),
                priority: if i == 0 { CommandPriority::High } else { CommandPriority::Normal },
            })?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum DispatchError {
    QueueFull,
    GpuBusy,
}

/// An async task that wraps a GPU command (simulates tokio::task).
pub struct GpuTask {
    id: u64,
    command: GpuCommand,
    state: TaskState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskState {
    Pending,
    Running,
    Completed,
    Failed,
}

impl GpuTask {
    pub fn new(id: u64, command: GpuCommand) -> Self {
        Self { id, command, state: TaskState::Pending }
    }

    pub fn execute(&mut self) -> GpuResult {
        self.state = TaskState::Running;
        let result = GpuResult {
            kernel_name: self.command.kernel_name.clone(),
            duration_us: 100,
            success: true,
            throughput_ops_s: 10000.0,
        };
        self.state = TaskState::Completed;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cmd(name: &str, prio: CommandPriority) -> GpuCommand {
        GpuCommand {
            kernel_name: name.into(),
            block_dim: (256, 1, 1),
            shared_mem: 0,
            submitted_at: Instant::now(),
            priority: prio,
        }
    }

    #[test]
    fn test_submit_and_execute() {
        let mut gpu = AsyncGpu::new(10);
        gpu.submit(make_cmd("attention", CommandPriority::Normal)).unwrap();
        assert_eq!(gpu.pending_count(), 1);
        let result = gpu.execute_one().unwrap();
        assert_eq!(result.kernel_name, "attention");
        assert!(result.success);
    }

    #[test]
    fn test_queue_full() {
        let mut gpu = AsyncGpu::new(2);
        gpu.submit(make_cmd("k1", CommandPriority::Normal)).unwrap();
        gpu.submit(make_cmd("k2", CommandPriority::Normal)).unwrap();
        assert!(gpu.submit(make_cmd("k3", CommandPriority::Normal)).is_err());
    }

    #[test]
    fn test_priority_execution() {
        let mut gpu = AsyncGpu::new(10);
        gpu.submit(make_cmd("low", CommandPriority::Low)).unwrap();
        gpu.submit(make_cmd("critical", CommandPriority::Critical)).unwrap();
        gpu.submit(make_cmd("normal", CommandPriority::Normal)).unwrap();

        let result = gpu.execute_by_priority().unwrap();
        assert_eq!(result.kernel_name, "critical");
    }

    #[test]
    fn test_batch_execution() {
        let mut gpu = AsyncGpu::new(100);
        for i in 0..50 {
            gpu.submit(make_cmd(&format!("kernel_{}", i), CommandPriority::Normal)).unwrap();
        }
        let results = gpu.execute_all();
        assert_eq!(results.len(), 50);
    }

    #[test]
    fn test_pipeline_submit() {
        let mut gpu = AsyncGpu::new(10);
        gpu.submit_pipeline(&["filter", "transform", "reduce"]).unwrap();
        assert_eq!(gpu.pending_count(), 3);
        let results = gpu.execute_all();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].kernel_name, "filter");
    }

    #[test]
    fn test_task_lifecycle() {
        let cmd = make_cmd("test", CommandPriority::High);
        let mut task = GpuTask::new(1, cmd);
        assert_eq!(task.state, TaskState::Pending);
        task.execute();
        assert_eq!(task.state, TaskState::Completed);
    }

    #[test]
    fn test_queue_utilization() {
        let mut gpu = AsyncGpu::new(10);
        assert_eq!(gpu.queue_utilization(), 0.0);
        for i in 0..5 { gpu.submit(make_cmd(&format!("k{}", i), CommandPriority::Normal)).unwrap(); }
        assert!((gpu.queue_utilization() - 0.5).abs() < 0.01);
    }
}

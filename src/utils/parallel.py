import datetime
import logging
import math
import multiprocessing as mp
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from logging.handlers import RotatingFileHandler

import psutil

IS_WINDOWS = platform.system() == "Windows"


if not IS_WINDOWS:
    import resource
else:
    class DummyResourceModule:
        RLIMIT_NOFILE = 8

        def getrlimit(self, resource_type):
            if resource_type == self.RLIMIT_NOFILE:
                return (512, 512)
            return (1024, 1024)

        def setrlimit(self, resource_type, limits):
            print(f"[WARN] resource.setrlimit not available on Windows. Requested limits: {limits}")
            pass

    resource = DummyResourceModule()


def increase_file_limit(desired_limit: int = 1024 * 10) -> None:
    if IS_WINDOWS:
        _increase_file_limit_windows(desired_limit)
    else:
        _increase_file_limit_macos_linux(desired_limit)


def _increase_file_limit_windows(desired_limit: int = 1024 * 10) -> None:
    print("[INFO] Skipping file limit increase on Windows system")
    return None


def _increase_file_limit_macos_linux(desired_limit: int = 1024 * 10) -> None:
    # Get current resource limits for file descriptors
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Current file descriptor limits: soft={soft_limit}, hard={hard_limit}")

    if desired_limit >= hard_limit:
        raise ValueError(f"Too large (requested: {desired_limit}, hard limit: {hard_limit})")

    # Determine new soft limit, bounded by system hard limit
    new_soft = min(desired_limit, hard_limit)

    # Only attempt to increase if current limit is insufficient
    if new_soft > soft_limit:
        # Apply the new resource limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))

        print(f"File descriptor limit increased to: {new_soft}")
    else:
        print(f"File descriptor limit already sufficient: {soft_limit}")

    return None


class RealTimeStreamProcessor:
    def __init__(self, logger, stream_name="stdout"):
        self.logger = logger
        self.stream_name = stream_name
        self.buffer = ""
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            if text:
                self.buffer += text
                lines = self.buffer.split("\n")

                self.buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        if self.stream_name == "stdout":
                            self.logger.info(f"[STDOUT] {line.strip()}")
                        else:  # stderr
                            self.logger.warning(f"[STDERR] {line.strip()}")

    def flush(self):
        with self.lock:
            if self.buffer.strip():
                if self.stream_name == "stdout":
                    self.logger.info(f"[STDOUT] {self.buffer.strip()}")
                else:
                    self.logger.warning(f"[STDERR] {self.buffer.strip()}")
                self.buffer = ""

    def close(self):
        self.flush()


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TaskConfig:
    task_id: int
    cores_per_task: int = 1
    problem_id: int | None = None
    algorithm_id: int | None = None


@dataclass
class TaskResult:
    task_id: int
    status: TaskStatus
    error_message: str | None
    duration: float


def setup_logger_for_task(task_id: int) -> logging.Logger:
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    utc_time = datetime.datetime.now(datetime.UTC)
    timestamp = utc_time.strftime("%Y%m%d_%H%M%S")

    # Create logger with task-specific name
    logger_name = f"task_{task_id}_utc_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to prevent duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler for task-specific log file
    handler = RotatingFileHandler(
        os.path.join("logs", f"{logger_name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
        delay=True,
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def set_thread_limits(cores_per_task: int = 1) -> None:
    # Environment variables for linear algebra libraries
    env_vars = {
        "OMP_NUM_THREADS": str(cores_per_task),
        "OPENBLAS_NUM_THREADS": str(cores_per_task),
        "MKL_NUM_THREADS": str(cores_per_task),
        "VECLIB_MAXIMUM_THREADS": str(cores_per_task),
        "NUMEXPR_NUM_THREADS": str(cores_per_task),
        "PYTHONUNBUFFERED": "1",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    # PyTorch thread configuration
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        torch.backends.cudnn.enabled = False
        torch.set_num_threads(cores_per_task)
        torch.set_num_interop_threads(cores_per_task)
    except ImportError as e:
        print(f"[WARN] Unexpected ImportError in set_thread_limits: {e}")
    except RuntimeError as e:
        if "after parallel" in str(e).lower():
            print(f"[WARN] Thread limits not applied (PyTorch already initialized): {e}")
        else:
            print(f"[WARN] Unexpected RuntimeError in set_thread_limits: {e}")
    except Exception as e:
        print(f"[WARN] Unexpected Error in set_thread_limits: {e}")


def run_task_with_resources(task_config: TaskConfig, task_func: Callable, **kwargs) -> TaskResult:
    start_time = time.time()

    # Setup logger for this task
    logger = setup_logger_for_task(task_config.task_id)
    logger.info(f"Starting task {task_config.task_id} with {task_config.cores_per_task} cores")

    # Execute the task function
    try:
        stdout_processor = RealTimeStreamProcessor(logger, "stdout")
        stderr_processor = RealTimeStreamProcessor(logger, "stderr")

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = stdout_processor
        sys.stderr = stderr_processor

        try:
            # Pass the task_config to the task function
            task_func(task_config, **kwargs)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            stdout_processor.flush()
            stderr_processor.flush()

        logger.info(f"Task {task_config.task_id} completed successfully")
        return TaskResult(
            task_id=task_config.task_id,
            status=TaskStatus.SUCCESS,
            error_message=None,
            duration=time.time() - start_time,
        )

    except Exception as e:
        # Capture and return any execution errors
        logger.error(f"Task {task_config.task_id} failed: {e}")
        return TaskResult(
            task_id=task_config.task_id,
            status=TaskStatus.FAILED,
            error_message=traceback.format_exc(),
            duration=time.time() - start_time,
        )

    finally:
        # Clean up logging handlers to prevent resource leaks
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        logging.getLogger().manager.loggerDict.pop(logger.name, None)


def parallel_execution_manager(
    task_func: Callable,
    task_configs: list[TaskConfig],
    max_cpu_ratio: float = 0.9,
    batch_delay: int = 10,
    **kwargs,
) -> None:
    """Execute multiple tasks in parallel with dedicated CPU cores for each task.

    Generic parallel execution manager that can handle both problem-based
    and algorithm-based parallelism. The function:
    1. Calculates optimal resource allocation
    2. Groups tasks into batches based on available cores
    3. Executes tasks in parallel with proper resource isolation
    4. Provides progress reporting and error handling

    Parameters
    ----------
    task_func : Callable
        Function that executes a single task. Must accept TaskConfig
        as first parameter.
    task_configs : list[TaskConfig]
        List of task configurations to process.
    max_cpu_ratio : float, default=0.9
        Maximum fraction of available CPU cores to utilize (0.0 to 1.0).
        Prevents complete system resource exhaustion.
    batch_delay : int, default=10
        Delay in seconds between batches for system stabilization and
        resource cleanup.

    Returns
    -------
    None

    Notes
    -----
    The function uses 'spawn' method for multiprocessing to ensure
    proper resource isolation. Each task runs in its own process with
    configured thread limits to prevent oversubscription.
    """
    # Validate task_configs
    if not task_configs:
        print("No tasks to execute.")
        return None

    # Validate uniform core allocation (simplified scheduler requirement)
    if not all(tc.cores_per_task == task_configs[0].cores_per_task for tc in task_configs):
        raise ValueError("All tasks must request the same number of cores for current scheduler.")

    cores_per_task = task_configs[0].cores_per_task if task_configs else 1
    if cores_per_task < 1:
        raise ValueError(f"cores_per_task must be >=1, got {cores_per_task}")

    # The total number of tasks
    num_tasks = len(task_configs)

    # Determine total available CPU cores
    logical_cores = psutil.cpu_count(logical=True)
    if logical_cores is None:
        raise ValueError("Failed to determine total CPU cores")

    # Calculate available cores based on user-defined ratio and reservations
    if not 0.0 < max_cpu_ratio <= 1.0:
        raise ValueError(f"max_cpu_ratio must be between 0.0 and 1.0, got {max_cpu_ratio}")

    set_thread_limits(cores_per_task)  # Reserve cores for main process
    available_cores = max(1, int(logical_cores * max_cpu_ratio) - cores_per_task)

    # Determine maximum number of concurrent tasks
    max_concurrent = max(1, available_cores // cores_per_task)

    # Don't create more processes than tasks
    processes = min(max_concurrent, num_tasks)

    # Calculate number of batches required
    num_batches = math.ceil(num_tasks / processes)

    # Display resource allocation information
    print("=" * 60)
    print("Parallel Execution Configuration:")
    print(f"  Total physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Total Logical cores: {psutil.cpu_count(logical=True)}")
    print(f"  Cores per task: {cores_per_task}")
    print(f"  Max concurrent tasks: {processes}")
    print(f"  Total tasks: {num_tasks}")
    print(f"  Number of batches: {num_batches}")
    print("=" * 60)

    # Configure for typical parallel processing workloads
    increase_file_limit(num_tasks * 50)

    # Process tasks in batches
    mp.set_start_method("spawn", force=True)
    for batch in range(num_batches):
        start_idx = batch * processes
        end_idx = min(start_idx + processes, num_tasks)
        batch_tasks = task_configs[start_idx:end_idx]

        print(f"\n{'=' * 40}")
        print(f"Batch {batch + 1}/{num_batches}")
        print(f"Processing {len(batch_tasks)} tasks")
        print(f"{'=' * 40}")

        # Create process pool for current batch
        with mp.Pool(
            processes=len(batch_tasks),
            initializer=set_thread_limits,
            initargs=(cores_per_task,),
            maxtasksperchild=1,
        ) as pool:
            # Create worker function with fixed resource configuration
            worker_func = partial(run_task_with_resources, task_func=task_func, **kwargs)

            # Execute tasks in parallel using unordered mapping
            results = pool.imap_unordered(worker_func, batch_tasks)

            # Process and report results
            success_count, fail_count = 0, 0
            results_list = list(results)
            results_list.sort(key=lambda r: r.task_id)
            for result in results_list:
                if result.status == "success":
                    success_count += 1
                else:
                    print(f"Task {result.task_id} failed: {result.error_message}")
                    fail_count += 1
            print(f"\nBatch summary: {success_count} succeeded, {fail_count} failed")

        # Add delay between batches for system stabilization
        if batch < num_batches - 1:  # No delay after last batch
            print(f"\nWaiting {batch_delay} seconds before next batch...")
            time.sleep(batch_delay)

    print("\n" + "=" * 60)
    print(f"All tasks completed: {num_tasks} total")
    print("=" * 60)

    return None


# ============================================================================
# Specialized functions for backward compatibility and convenience
# ============================================================================


def problem_based_run_with_fixed_resources(
    run_algorithms_for_problem: Callable,
    *,
    problem_size: int,
    cores_per_task: int = 1,
    max_cpu_ratio: float = 0.9,
    **kwargs,
) -> None:
    # Create task configurations for each problem
    task_configs = [TaskConfig(task_id=i, problem_id=i, cores_per_task=cores_per_task) for i in range(problem_size)]

    # Execute in parallel
    parallel_execution_manager(
        task_func=run_algorithms_for_problem,
        task_configs=task_configs,
        max_cpu_ratio=max_cpu_ratio,
        **kwargs,
    )


def algorithm_based_run_with_fixed_resources(
    run_problems_for_algorithm: Callable,
    *,
    algorithm_size: int,
    cores_per_task: int = 1,
    max_cpu_ratio: float = 0.9,
    **kwargs,
) -> None:
    # Create task configurations for each algorithm
    task_configs = [TaskConfig(task_id=i, algorithm_id=i, cores_per_task=cores_per_task) for i in range(algorithm_size)]

    # Execute in parallel
    parallel_execution_manager(
        task_func=run_problems_for_algorithm,
        task_configs=task_configs,
        max_cpu_ratio=max_cpu_ratio,
        **kwargs,
    )


def combined_parallel_run(
    run_combination: Callable,
    *,
    problem_size: int,
    algorithm_size: int,
    cores_per_task: int = 1,
    max_cpu_ratio: float = 0.9,
    **kwargs,
) -> None:
    task_configs = []
    task_id = 0

    # Create configurations for all combinations
    for problem_id in range(problem_size):
        for algorithm_id in range(algorithm_size):
            task_configs.append(
                TaskConfig(
                    task_id=task_id,
                    problem_id=problem_id,
                    algorithm_id=algorithm_id,
                    cores_per_task=cores_per_task,
                )
            )
            task_id += 1

    print(f"Total combinations to run: {len(task_configs)}")
    print(f"(Problems: {problem_size} x Algorithms: {algorithm_size} = {len(task_configs)})")

    # Execute in parallel
    parallel_execution_manager(
        task_func=run_combination,
        task_configs=task_configs,
        max_cpu_ratio=max_cpu_ratio,
        **kwargs,
    )


def serial_run(
    run_combination: Callable,
    *,
    problem_size: int,
    algorithm_size: int,
    cores_per_task: int = 1,
    max_cpu_ratio: float = 0.9,
    **kwargs,
) -> None:
    task_configs = []
    task_id = 0

    # Create configurations for all combinations
    for problem_id in range(problem_size):
        for algorithm_id in range(algorithm_size):
            task_configs.append(
                TaskConfig(
                    task_id=task_id,
                    problem_id=problem_id,
                    algorithm_id=algorithm_id,
                    cores_per_task=cores_per_task,
                )
            )
            task_id += 1

    print(f"Total combinations to run: {len(task_configs)}")
    print(f"(Problems: {problem_size} x Algorithms: {algorithm_size} = {len(task_configs)})")

    # Execute in serial
    for task_config in task_configs:
        parallel_execution_manager(
            task_func=run_combination,
            task_configs=[task_config],
            max_cpu_ratio=max_cpu_ratio,
            **kwargs,
        )

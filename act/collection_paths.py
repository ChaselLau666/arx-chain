"""Dataset path helpers for collection sessions."""

from pathlib import Path


def normalize_task_name(task_name):
    """Return a safe task-directory name."""
    task_name = str(task_name).strip()
    if not task_name:
        raise ValueError('must not be empty')
    if task_name in {'.', '..'} or Path(task_name).name != task_name:
        raise ValueError('must be a single directory name, not a path')
    return task_name


def task_dataset_dir(dataset_root, task_name):
    """Return the per-task dataset directory, rejecting path-like task names."""
    return Path(dataset_root) / normalize_task_name(task_name)

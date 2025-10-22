"""
Utility functions for multi-GPU distributed training.
This module provides helper functions for DDP training setup and data loading.
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_for_distributed(is_master):
    """
    Disable printing when not in master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get the number of processes in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save checkpoint only on the master process."""
    if is_main_process():
        torch.save(*args, **kwargs)


def get_distributed_sampler(dataset, num_replicas=None, rank=None, shuffle=True):
    """
    Create a DistributedSampler for the dataset.
    
    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: Whether to shuffle the data
    
    Returns:
        DistributedSampler instance
    """
    if num_replicas is None:
        num_replicas = get_world_size()
    if rank is None:
        rank = get_rank()
    
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=True
    )


def get_distributed_dataloader(dataset, batch_size, num_workers=4, shuffle=True, pin_memory=True):
    """
    Create a DataLoader with DistributedSampler for multi-GPU training.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers per GPU
        shuffle: Whether to shuffle the data
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance with DistributedSampler
    """
    sampler = get_distributed_sampler(dataset, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


def reduce_dict(input_dict, average=True):
    """
    Reduce values in a dictionary across all processes.
    
    Args:
        input_dict: Dictionary with tensors to reduce
        average: If True, average the values; otherwise sum them
    
    Returns:
        Dictionary with reduced values
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


def reduce_tensor(tensor, average=True):
    """
    Reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: If True, average the values; otherwise sum them
    
    Returns:
        Reduced tensor
    """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    
    with torch.no_grad():
        dist.all_reduce(tensor)
        if average:
            tensor /= world_size
    
    return tensor


def gather_tensors(tensor):
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
    
    Returns:
        List of tensors from all processes (only valid on rank 0)
    """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    
    # Get tensor size
    local_size = torch.tensor([tensor.size(0)], device=tensor.device)
    size_list = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    
    # Pad tensor to max size
    if local_size.item() < max_size:
        padding = torch.zeros(
            (max_size - local_size.item(),) + tensor.size()[1:],
            dtype=tensor.dtype,
            device=tensor.device
        )
        tensor = torch.cat([tensor, padding], dim=0)
    
    # Gather tensors
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # Trim to original sizes
    tensor_list = [tensor[:size] for tensor, size in zip(tensor_list, size_list)]
    
    return tensor_list


class MetricLogger:
    """
    Logger for metrics during distributed training.
    Handles metric aggregation across multiple processes.
    """
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def get_dict(self):
        return {k: v.avg for k, v in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def synchronize_between_processes(self):
        """Synchronize metrics across all processes."""
        if not is_dist_avail_and_initialized():
            return
        
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count

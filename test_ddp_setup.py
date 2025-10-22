#!/usr/bin/env python3
"""
Quick test script to verify multi-GPU training setup.
This script performs a short training run to ensure everything is configured correctly.

Usage:
    torchrun --nproc_per_node=2 test_ddp_setup.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    dist.barrier()
    
    return True, rank, world_size, local_rank


def test_basic_ddp():
    """Test basic DDP functionality."""
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    if not is_distributed:
        print("❌ Distributed training not available.")
        print("Please run with: torchrun --nproc_per_node=NUM_GPUS test_ddp_setup.py")
        return False
    
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"✅ Distributed training initialized successfully!")
        print(f"   World size: {world_size}")
        print(f"   Available GPUs: {torch.cuda.device_count()}")
    
    # Test 1: Basic tensor operations
    if rank == 0:
        print("\n[Test 1] Testing tensor operations on each GPU...")
    
    test_tensor = torch.randn(10, 10).to(device)
    result = test_tensor @ test_tensor.T
    
    if rank == 0:
        print(f"   ✅ Tensor operations work on GPU {local_rank}")
    
    dist.barrier()
    
    # Test 2: Communication between GPUs
    if rank == 0:
        print("\n[Test 2] Testing communication between GPUs...")
    
    value = torch.tensor([rank * 1.0], device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(world_size))
    
    if abs(value.item() - expected_sum) < 1e-5:
        if rank == 0:
            print(f"   ✅ Communication works! Sum = {value.item()}")
    else:
        if rank == 0:
            print(f"   ❌ Communication failed. Expected {expected_sum}, got {value.item()}")
        return False
    
    dist.barrier()
    
    # Test 3: Simple model with DDP
    if rank == 0:
        print("\n[Test 3] Testing DDP model wrapper...")
    
    model = torch.nn.Linear(100, 50).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Forward pass
    dummy_input = torch.randn(32, 100).to(device)
    output = ddp_model(dummy_input)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    if rank == 0:
        print(f"   ✅ DDP model wrapper works!")
        print(f"   Model output shape: {output.shape}")
        print(f"   Loss: {loss.item():.4f}")
    
    dist.barrier()
    
    # Test 4: Gradient synchronization
    if rank == 0:
        print("\n[Test 4] Testing gradient synchronization...")
    
    # Get a parameter's gradient
    param = next(ddp_model.parameters())
    local_grad_norm = param.grad.norm().item()
    
    # All ranks should have the same gradient after backward
    grad_tensor = torch.tensor([local_grad_norm], device=device)
    all_grads = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
    dist.all_gather(all_grads, grad_tensor)
    
    all_grads_values = [g.item() for g in all_grads]
    grads_match = all(abs(g - all_grads_values[0]) < 1e-5 for g in all_grads_values)
    
    if grads_match:
        if rank == 0:
            print(f"   ✅ Gradients are synchronized across all GPUs!")
            print(f"   Gradient norm: {local_grad_norm:.6f}")
    else:
        if rank == 0:
            print(f"   ❌ Gradients are not synchronized!")
            print(f"   Gradients: {all_grads_values}")
        return False
    
    dist.barrier()
    
    # Cleanup
    dist.destroy_process_group()
    
    if rank == 0:
        print("\n" + "="*50)
        print("✅ All tests passed!")
        print("Your multi-GPU setup is ready for training.")
        print("="*50)
    
    return True


def test_pytorch_installation():
    """Test if PyTorch is installed correctly with CUDA support."""
    print("\n" + "="*50)
    print("Testing PyTorch Installation")
    print("="*50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please install PyTorch with CUDA support.")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    print("✅ PyTorch is installed correctly with CUDA support!")
    return True


if __name__ == "__main__":
    # First test PyTorch installation
    if not test_pytorch_installation():
        sys.exit(1)
    
    # Then test DDP setup
    if 'RANK' in os.environ:
        # Running with torchrun
        success = test_basic_ddp()
        sys.exit(0 if success else 1)
    else:
        # Not running with torchrun
        print("\n" + "="*50)
        print("DDP Test Instructions")
        print("="*50)
        print("To test multi-GPU training, run:")
        print(f"  torchrun --nproc_per_node=2 {sys.argv[0]}")
        print("\nReplace 2 with the number of GPUs you want to test.")
        print("="*50)

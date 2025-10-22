# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model with multi-GPU support using DistributedDataParallel (DDP).

Usage:
    # Single node, multiple GPUs:
    torchrun --nproc_per_node=4 train/train_mdm_ddp.py --save_dir ./save/my_experiment --dataset humanml

    # Multiple nodes (on each node run):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=<PORT> train/train_mdm_ddp.py --save_dir ./save/my_experiment --dataset humanml
"""

import os
import json
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_ddp import TrainLoopDDP
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform


def setup_distributed():
    """
    Initialize the distributed environment.
    This function expects torchrun to set the environment variables.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    dist.barrier()
    
    return True, rank, world_size, local_rank


def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = train_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    if not is_distributed:
        print("Error: This script requires distributed training setup.")
        print("Please run with torchrun:")
        print("  torchrun --nproc_per_node=NUM_GPUS train/train_mdm_ddp.py [args]")
        return
    
    # Set device for this process
    args.device = local_rank
    device = torch.device(f'cuda:{local_rank}')
    
    # Fix seed (with different seed per rank for data loading diversity)
    fixseed(args.seed + rank)
    
    # Only rank 0 handles logging and directory creation
    if rank == 0:
        train_platform_type = eval(args.train_platform_type)
        train_platform = train_platform_type(args.save_dir)
        train_platform.report_args(args, name='Args')

        if args.save_dir is None:
            raise FileNotFoundError('save_dir was not specified.')
        elif os.path.exists(args.save_dir) and not args.overwrite:
            raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
        elif not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        args_path = os.path.join(args.save_dir, 'args.json')
        with open(args_path, 'w') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
        
        print(f"Training with {world_size} GPUs")
        print(f"Global batch size: {args.batch_size * world_size}")
    else:
        train_platform_type = NoPlatform
        train_platform = train_platform_type(args.save_dir)
    
    # Wait for rank 0 to create directories
    dist.barrier()
    
    # Setup dist_util
    dist_util.setup_dist(args.device)

    if rank == 0:
        print("Creating data loader...")

    # Create dataset loader with distributed sampler
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        fixed_len=args.pred_len + args.context_len,
        pred_len=args.pred_len,
        device=device,
        use_cache=args.use_cache
    )
    
    # Wrap with DistributedSampler for multi-GPU training
    # Note: The get_dataset_loader function should be modified to accept a sampler parameter
    # or we need to recreate the DataLoader with DistributedSampler
    # For now, we'll wrap the existing loader
    if hasattr(data, 'dataset'):
        sampler = DistributedSampler(
            data.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        
        # Recreate DataLoader with distributed sampler
        from torch.utils.data import DataLoader
        data = DataLoader(
            data.dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=data.num_workers if hasattr(data, 'num_workers') else 4,
            pin_memory=True,
            drop_last=True
        )

    if rank == 0:
        print("Creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(args, data)
    model = model.to(device)
    model.rot2xyz.smpl_model.eval()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0
        print(f'Total params: {total_params:.2f}M')
        print("Training...")
    
    # Train with DDP
    TrainLoopDDP(args, train_platform, model, diffusion, data).run_loop()
    
    if rank == 0:
        train_platform.close()
    
    cleanup()


if __name__ == "__main__":
    main()

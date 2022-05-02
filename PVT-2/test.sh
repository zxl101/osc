python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset CUB --resume ./checkpoints/efficientnet_b0/CUB/checkpoint.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset SCAR --resume ./checkpoints/efficientnet_b0/SCAR/checkpoint.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset CIFAR10 --resume ./checkpoints/efficientnet_b0/CIFAR10/checkpoint.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset CIFAR100 --resume ./checkpoints/efficientnet_b0/CIFAR100/checkpoint.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/swin.py --data-path ./ --dataset CIFAR100 --resume ./checkpoints/swin/CIFAR100/checkpoint.pth --eval

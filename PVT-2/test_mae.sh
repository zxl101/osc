#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset CUB --resume ./checkpoints/MAE/CUB/checkpoint.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset SCAR --resume ./checkpoints/MAE/SCAR/checkpoint.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset CIFAR10 --resume ./checkpoints/MAE/CIFAR10/checkpoint.pth --eval

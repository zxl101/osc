#MAE
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --finetune ./mae_cub.pth --dataset CUB

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --finetune ./mae_scar.pth --dataset SCAR

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --finetune ./mae_cifar10.pth --dataset CIFAR10

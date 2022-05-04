#sh dist_train.sh configs/pvt/pvt_small.py 1 --data-path ./ --resume ./ --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wclu 0 --lr 0.0001

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CIFAR10

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CIFAR100

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/resnet50.py --data-path ./ --dataset CUB

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/resnet50.py --data-path ./ --dataset SCAR

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/resnet50.py --data-path ./ --dataset CIFAR10

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/resnet50.py --data-path ./ --dataset CIFAR100

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/swin.py --data-path ./ --dataset CUB

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/swin.py --data-path ./ --dataset SCAR

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/swin.py --data-path ./ --dataset CIFAR10

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/swin.py --data-path ./ --dataset CIFAR100

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset CUB

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset SCAR

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset CIFAR10

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/efficientnet_b0.py --data-path ./ --dataset CIFAR100

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/vit_moco.py --data-path ./ --finetune ./checkpoint_0214.pth.tar

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/pvt_small_dgrl.py --data-path ./ --finetune ./pvt_small.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/vit_ssl.py --data-path ./ --finetune ./vit-s-300ep-sn.pth.tar

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/pvt_small_aux.py --data-path ./ --finetune ./pvt_small.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main3.py --config configs/pvt/pvt_small_tree.py --data-path ./ --finetune ./pvt_small.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/pvt_small.py --data-path ./ --finetune ./pvt_small.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/pvt_small_ce.py --data-path ./ --finetune ./pvt_small.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/pvt_small_clu.py --data-path ./ --finetune ./pvt_small.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt/pvt_small.py --data-path ./  
    
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt/pvt_small.py --data-path ./ --finetune ./checkpoints/pvt_small/jigsaw_checkpoint.pth
    
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b1.py --data-path ./ --finetune ./pvt_v2_b1.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b2.py --data-path ./ --finetune ./pvt_v2_b2.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b3.py --data-path ./ --finetune ./pvt_v2_b3.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b4.py --data-path ./ --finetune ./pvt_v2_b4.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b5.py --data-path ./ --finetune ./pvt_v2_b5.pth

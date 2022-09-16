#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50 --finetune ./ssl_pretrained_imagenet/mae_nabird.pth  --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset NABird --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50 --finetune ./ssl_pretrained_imagenet/mae_nabird.pth  --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49 --finetune ./ssl_pretrained_imagenet/mae_nabird.pth  --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 0 --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50 --finetune ./ssl_pretrained_imagenet/mae_pretrain.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset NABird --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 0 --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50 --finetune ./ssl_pretrained_imagenet/mae_pretrain.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 0 --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49 --finetune ./ssl_pretrained_imagenet/mae_pretrain.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0.1 --lr 0.0001 --ce_warmup_epoch 0 --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50 --finetune ./checkpoint.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset NABird --wce 1 --wclu 0.1 --lr 0.0001 --ce_warmup_epoch 0 --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50 --finetune ./checkpoint.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --wce 1 --wclu 0.1 --lr 0.0001 --ce_warmup_epoch 0 --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49 --finetune ./checkpoint.pth --epochs 200


#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49 --finetune ./ssl_pretrained_imagenet/dino_pretrain.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49 --finetune ./checkpoint.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CIFAR10 --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 5 --num_aux_classes 2 --num_test_classes 3 --finetune ./ssl_pretrained_imagenet/dino_pretrain.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CIFAR10 --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 5 --num_aux_classes 2 --num_test_classes 3 --finetune ./checkpoint.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CIFAR100 --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 50 --num_aux_classes 25 --num_test_classes 25 --finetune ./ssl_pretrained_imagenet/dino_pretrain.pth --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CIFAR100 --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 50 --num_aux_classes 25 --num_test_classes 25 --finetune ./checkpoint.pth --epochs 200


CUDA_VISIBLE_DEVICES=1 python main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50 --epochs 200

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main5.py --config configs/backbones/vit_base.py --data-path ./ --dataset NABird --wce 1 --wclu 0 --wre 0.5 --lr 0.0001 --ce_warmup_epoch 200 --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50 --epochs 200 --finetune ./ssl_pretrained_imagenet/dino_nabird.pth




#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 100 --wclu 0 --lr 0.0001 --ce_warmup_epoch 0 --finetune imagenet_pretrained.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 30 --wclu 0.5 --lr 0.0001 --ce_warmup_epoch 0 --finetune imagenet_pretrained.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0 --lr 0.0001 --ce_warmup_epoch 0 --finetune mae_finetuned_vit_base.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 0 --wclu 1 --lr 0.0001 --ce_warmup_epoch 0 --finetune mae_finetuned_vit_base.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 1 --lr 0.0001 --ce_warmup_epoch 50 --finetune mae_finetuned_vit_base.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 1 --lr 0.001 --ce_warmup_epoch 50 --finetune mae_finetuned_vit_base.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0.01 --lr 0.0001 --ce_warmup_epoch 50 --finetune mae_finetuned_vit_base.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0 --lr 0.00003 --ce_warmup_epoch 0

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 1 --lr 0.00003 --ce_warmup_epoch 0



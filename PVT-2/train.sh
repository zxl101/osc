python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 0 --wclu 0.001 --lr 0.0000047 --ce_warmup_epoch 0 --finetune mae_finetuned_vit_base.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 0 --lr 0.00003 --ce_warmup_epoch 0

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --wce 1 --wclu 1 --lr 0.00003 --ce_warmup_epoch 0



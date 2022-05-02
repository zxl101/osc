#MAE
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset CUB --finetune ./ssl_pretrained_imagenet/mae_pretrain.pth --eval

#DINO
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset CUB --finetune ./ssl_pretrained_imagenet/dino_pretrain.pth --eval



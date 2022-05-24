python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_ts_sss.py --data-path ./ --dataset CUB --finetune ./dino_pretrained/CUB_ts_sss/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_tss_ss.py --data-path ./ --dataset CUB --finetune ./dino_pretrained/CUB_tss_ss/checkpoint.pth --eval

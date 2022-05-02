#DINO
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_ts_sss.py --data-path ./ --dataset CUB --finetune ./dino_cub_ts_sss.pth --eval
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_tss_ss.py --data-path ./ --dataset CUB --finetune ./dino_cub_tss_ss.pth --eval
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_ts_sss.py --data-path ./ --dataset SCAR --finetune ./dino_scar_ts_sss.pth --eval

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_ts_ss.py --data-path ./ --dataset CUB --finetune ./dino_pretrained/CUB_ts_ss/checkpoint.pth --eval
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_ts_sss.py --data-path ./ --dataset CUB --finetune ./dino_pretrained/freeze/CUB_ts_sss/checkpoint.pth --eval
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_tss_ss.py --data-path ./ --dataset CUB --finetune ./dino_pretrained/freeze/CUB_tss_ss/checkpoint.pth --eval
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_tss_sss.py --data-path ./ --dataset CUB --finetune ./dino_pretrained/freeze/CUB_tss_sss/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_ts_sss.py --data-path ./ --dataset SCAR --finetune ./dino_pretrained/SCAR_ts_sss/checkpoint.pth --eval
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/t_s/DINO_tss_ss.py --data-path ./ --dataset SCAR --finetune ./dino_pretrained/SCAR_tss_ss/checkpoint.pth --eval

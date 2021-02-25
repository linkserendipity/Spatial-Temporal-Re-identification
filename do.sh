# CUDA_VISIBLE_DEVICES=0 python3 train_market.py --PCB  --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "/home/ls/raw-dataset/dataset/market_rename/"
# CUDA_VISIBLE_DEVICES=0 python3 test_st_market.py --PCB --name ft_ResNet50_pcb_market_e --test_dir "/home/ls/raw-dataset/dataset/market_rename/"

CUDA_VISIBLE_DEVICES=0 python3 gen_st_model_market.py --name ft_ResNet50_pcb_market_e --data_dir "/home/ls/raw-dataset/dataset/market_rename/"
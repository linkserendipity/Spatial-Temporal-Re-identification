#  python3 prepare.py --Market
# CUDA_VISIBLE_DEVICES=1 python3 train_market.py --batchsize 64 --PCB --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir /mnt/SSD/ls/dataset_st/market_rename
# CUDA_VISIBLE_DEVICES=1 python3 test_st_market.py --batchsize 32 --PCB --name ft_ResNet50_pcb_market_e --test_dir /mnt/SSD/ls/dataset_st/market_rename
## 19732 3368
# CUDA_VISIBLE_DEVICES=1 python3 gen_st_model_market.py --name ft_ResNet50_pcb_market_e --data_dir /mnt/SSD/ls/dataset_st/market_rename
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_st.py --name ft_ResNet50_pcb_market_e
#*rerank
# CUDA_VISIBLE_DEVICES=1 python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_market_e
python3 evaluate_rerank_market.py --name ft_ResNet50_pcb_market_e
#todo add log time!
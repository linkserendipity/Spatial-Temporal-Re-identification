# python3 prepare.py  --Duke 
# CUDA_VISIBLE_DEVICES=1 python3 train_duke.py --batchsize 64 --PCB --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "/mnt/SSD/ls/dataset_st/DukeMTMC_prepare/"
# CUDA_VISIBLE_DEVICES=1 python3 test_st_duke.py --batchsize 32 --PCB --name ft_ResNet50_pcb_duke_e --test_dir "/mnt/SSD/ls/dataset_st/DukeMTMC_prepare/"
##17661 2228
# CUDA_VISIBLE_DEVICES=1 python3 gen_st_model_duke.py --name ft_ResNet50_pcb_duke_e --data_dir "/mnt/SSD/ls/dataset_st/DukeMTMC_prepare/"
# CUDA_VISIBLE_DEVICES=1 python3 evaluate_st.py --name ft_ResNet50_pcb_duke_e
#*rerank
# CUDA_VISIBLE_DEVICES=1 python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_duke_e
python3 evaluate_rerank_duke.py --name ft_ResNet50_pcb_duke_e
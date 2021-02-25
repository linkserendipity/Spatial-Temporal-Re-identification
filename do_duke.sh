CUDA_VISIBLE_DEVICES=3 python3 train_duke.py --PCB --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "/home/ls/raw-dataset/dataset/DukeMTMC_prepare/"
# CUDA_VISIBLE_DEVICES=1 python3 test_st_duke.py --PCB --name ft_ResNet50_pcb_duke_e --data_dir "/home/ls/raw-dataset/dataset/DukeMTMC_prepare/"

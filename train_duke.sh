# python3 prepare.py  --Duke 
# python3 train_duke.py --PCB --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "/home/ccc/Link/data/dataset/DukeMTMC_prepare"
# python3 test_st_duke.py --PCB --name ft_ResNet50_pcb_duke_e --test_dir "/home/ccc/Link/data/dataset/DukeMTMC_prepare"
# python3 gen_st_model_duke.py --name ft_ResNet50_pcb_duke_e --data_dir "/home/ccc/Link/data/dataset/DukeMTMC_prepare"
# python3 evaluate_st.py --name ft_ResNet50_pcb_duke_e
#rerank
# python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_duke_e
# python3 evaluate_rerank_duke.py --name ft_ResNet50_pcb_duke_e
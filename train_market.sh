## python3 prepare.py --Market
# python3 train_market.py --PCB --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "/home/ccc/Link/data/dataset/market_rename/"
# python3 test_st_market.py --PCB --name ft_ResNet50_pcb_market_e --test_dir "/home/ccc/Link/data/dataset/market_rename/"
# python3 gen_st_model_market.py --name ft_ResNet50_pcb_market_e --data_dir "/home/ccc/Link/data/dataset/market_rename/"
# python3 evaluate_st.py --name ft_ResNet50_pcb_market_e
#rerank
# python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_market_e
# python3 evaluate_rerank_market.py --name ft_ResNet50_pcb_market_e
#todo add log time!
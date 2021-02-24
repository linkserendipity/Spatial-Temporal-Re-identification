# rank and mAP

## python3 evaluate_st.py --name ft_ResNet50_pcb_market_e 

top1:0.977435 top5:0.992577 top10:0.994359 **mAP:0.876665**

alpha,smooth: 5 50

## evaluate_rerank_market.py
```
python3 evaluate_rerank_market.py --name ft_ResNet50_pcb_market_e
all_dist shape: (23100, 23100)
query_cam shape: (3368,)
calculate initial distance
all_dist shape: (23100, 23100)
initial_rank shape: (23100, 23100)
Reranking complete in 0m 54s
```
top1:0.971200 top5:0.988717 top10:0.993468 **mAP:0.934024**

---

## python3 evaluate_st.py --name ft_ResNet50_pcb_duke_e 
top1:0.943447 top5:0.974865 top10:0.982495 mAP:0.840928

alpha,smooth: 5 50

## evaluate_rerank_duke.py






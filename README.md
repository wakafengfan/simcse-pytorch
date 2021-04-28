# simcse-pytorch

最近出圈的无监督语义表示模型simcse，依然是基于苏神的keras版本改造的pytorch版本，
先占坑待后续补充更多实验，并补充Danqi女神的pytorch版本在中文上效果

目前仅实验了roberta-wwm在LCQMC上无监督训练效果，评测指标是Spearman correlation


| Model                      | correlation score | 
| -------------------------- | ----------------- | 
| `roberta-wwm`              | 0.67029           | 
| dropout_rate=0.1           |                   | 
| learning_rate=1e-5         |                   |
| pooling: first-last-avg    |                   | 
| `roberta-wwm(no training)` | 0.60377           | 
| pooling: first-last-avg    |                   | 


### 参考
SimCSE: Simple Contrastive Learning of Sentence Embeddings https://arxiv.org/pdf/2104.08821.pdf
https://kexue.fm/archives/8348
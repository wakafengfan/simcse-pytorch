import logging
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

# data
data_dir = Path(ROOT_PATH)/ "data"
model_dir = Path(ROOT_PATH) / "model"

bert_data_path = Path.home() / 'db__pytorch_pretrained_bert'
bert_vocab_path = bert_data_path / 'bert-base-chinese' / 'vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese'
uer_bert_base_model_path = bert_data_path / 'uer-bert-base'
uer_bert_large_model_path = bert_data_path / 'uer-bert-large'

tencent_w2v_path = Path.home() / 'db__word2vec'

roberta_large_model_path = bert_data_path / 'chinese_Roberta_bert_wwm_large_ext_pytorch'

bert_insurance_path = bert_data_path / 'bert_insurance_v2'

bert_wwm_path = bert_data_path / "chinese_wwm_ext_L-12_H-768_A-12"
bert_wwm_pt_path = bert_data_path / "chinese_wwm_ext_pytorch"
robert_wwm_pt_path = bert_data_path / "chinese_roberta_wwm_ext_pytorch"

mt5_pt_path = bert_data_path / "mt5_small_pt"
nezha_pt_path = bert_data_path / "nezha-cn-base"

simbert_path = bert_data_path / "chinese_simbert_L-12_H-768_A-12"
simbert_pt_path = bert_data_path / "chinese_simbert_pt"

common_data_path = Path.home() / 'db__common_dataset'
open_dataset_path = common_data_path / "open_dataset"




###############################################
# log
###############################################

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'begin progress ...')



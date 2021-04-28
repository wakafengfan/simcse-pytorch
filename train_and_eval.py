import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, AutoConfig

from bojone_snippets import DataGenerator, sequence_padding
from bojone_tokenizers import Tokenizer
from configuration.config import *
from opt import create_optimizer_and_scheduler
from utils import l2_normalize, compute_corrcoef

batch_size = 64
maxlen = 64
task_name = "LCQMC"
epochs = 1
gradient_accumulation_steps = 1


# 加载数据
def load_data(data_path):
    D = []
    for line in data_path.open():
        text1, text2, label = line.strip().split("\t")
        D.append((text1, text2, float(label)))
    return D


# 加载分词器
dict_path = str(robert_wwm_pt_path / "vocab.txt")
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text, in self.sample(random):
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            if "mode" in self.kwargs and self.kwargs["mode"] == "train":
                batch_token_ids.append(token_ids)
                batch_segment_ids.append([1] * len(token_ids))
            batch_segment_ids.append([1] * len(token_ids))

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long)
                batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long)
                yield batch_token_ids, batch_segment_ids
                batch_token_ids, batch_segment_ids = [], []


class EncodingModel(BertPreTrainedModel):
    def __init__(self, config):
        super(EncodingModel, self).__init__(config)
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask, encoder_type="fist-last-avg"):
        """

        :param input_ids:
        :param attention_mask:
        :param encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        """

        output = self.bert(input_ids, attention_mask, output_hidden_states=True)

        if encoder_type == "fist-last-avg":
            first = output.hidden_states[1]  # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)
            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [b,d]
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [b,d]
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1,2), kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type == "last-avg":
            sequence_output = output.last_hidden_state  # [b,s,d]
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1,2), kernel_size=seq_length).squeeze(-1)  # [b,d]
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output


def convert_to_ids(data):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels


def split_data(dat):
    a_texts, b_texts, labels = [],[],[],
    for d in tqdm(dat):
        a_texts.append(d[0])
        b_texts.append(d[1])
        labels.append(d[2])
    return a_texts, b_texts, labels


datasets = {fn: load_data(open_dataset_path / task_name / f"{fn}.tsv") for fn in ["train", "dev", "test"]}
all_weights, all_texts, all_labels = [], [], []
train_texts = []
for name, data in datasets.items():
    a_texts, b_texts, labels = split_data(data)
    all_weights.append(len(data))
    all_texts.append((a_texts, b_texts))
    all_labels.append(labels)

    train_texts.extend(a_texts)
    train_texts.extend(b_texts)

np.random.shuffle(train_texts)
train_texts = train_texts[:10000]
train_generator = data_generator(train_texts, batch_size, mode="train")


# 计算loss
loss_func = nn.BCEWithLogitsLoss()
def simcse_loss(y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = torch.arange(0, y_pred.size(0))  # [b]

    idxs_1 = idxs[None, :]  # [1,b]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]  # [b,1]
    y_true = idxs_1 == idxs_2
    y_true = y_true.to(torch.float).to(device)
    # 计算相似度
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = torch.matmul(y_pred, y_pred.transpose(0,1))  # [b,d] * [b.d] -> [b,1]
    similarities = similarities - torch.eye(y_pred.size(0)).to(device) * 1e12
    similarities = similarities * 20
    loss = loss_func(similarities, y_true)
    return loss


# 加载模型
config_path = robert_wwm_pt_path / "bert_config.json"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config_path, hidden_dropout_prob=0.1)
model = EncodingModel.from_pretrained(robert_wwm_pt_path, config=config)

optimizer, scheduler = create_optimizer_and_scheduler(model=model, lr=1e-5, num_training_steps=train_generator.steps * epochs // gradient_accumulation_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# train
model.zero_grad()
for e in range(epochs):
    model.train()
    for step, batch in enumerate(train_generator):
        # if step > 1: break
        batch = [_.to(device) for _ in batch]
        input_ids, seg_ids = batch
        encoding_output = model(input_ids, seg_ids)

        loss = simcse_loss(encoding_output)
        loss.backward()

        if step % gradient_accumulation_steps == 0 and step != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 100 == 0 and step != 0:
            print(f"epoch: {e} - batch: {step}/{train_generator.steps} - loss: {loss}")

model.eval()

# 语料向量化
all_vecs = []
for a_texts, b_texts in all_texts:
    a_text_generator = data_generator(a_texts, batch_size, mode="eval")
    b_text_generator = data_generator(b_texts, batch_size, mode="eval")

    all_a_vecs = []
    for eval_batch in tqdm(a_text_generator):
        eval_batch = [_.to(device) for _ in eval_batch]
        with torch.no_grad():
            eval_encodings = model(*eval_batch)
            eval_encodings = eval_encodings.cpu().detach().numpy()
            all_a_vecs.extend(eval_encodings)

    all_b_vecs = []
    for eval_batch in tqdm(b_text_generator):
        eval_batch = [_.to(device) for _ in eval_batch]
        with torch.no_grad():
            eval_encodings = model(*eval_batch)
            eval_encodings = eval_encodings.cpu().detach().numpy()
            all_b_vecs.extend(eval_encodings)

    all_vecs.append((np.array(all_a_vecs), np.array(all_b_vecs)))


# 标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

print(all_corrcoefs)





import pandas as pd

import torch
import torch.nn as nn
import logging
import sys
import joblib
import logging
import json
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.layers import Input, Dense, LSTM, Multiply, Dropout
from sklearn.model_selection import train_test_split
import re
import warnings
from ded import beam_search as bs
from ded.arguments import parse_args
from ded import utils
import warnings
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

SEED = 42  # You can change this to any fixed value

import random

# Python's built-in random module
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Onexlstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):  # , num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Project input to hidden_size if dimensions differ
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

        # Configure mLSTM layer
        mlstm_layer_config = mLSTMLayerConfig(
            conv1d_kernel_size=4,  # Controls convolutional feature mixing
            qkv_proj_blocksize=4,  # Projection block size for QKV transformations
            num_heads=4  # Number of parallel memory heads
        )

        # mLSTM block configuration
        mlstm_block_config = mLSTMBlockConfig(mlstm=mlstm_layer_config)

        slstm_block_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent"
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
        )

        # xLSTM stack configuration
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block_config,
            slstm_block=slstm_block_config,
            context_length=256,  # Maximum sequence length expected
            num_blocks=num_layers,  # Number of mLSTM layers
            embedding_dim=hidden_size,  # Same as LSTM's hidden_size
            slstm_at=[]  # Empty list = use only mLSTM blocks
        )

        # Create xLSTM stack with mLSTM blocks
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)

        # Regularization and output layer
        self.dropout = nn.Dropout(0.3)
        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Ensure correct input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Project input features to hidden dimension
        x = self.input_proj(x)

        # Forward pass through mLSTM blocks
        out = self.xlstm_stack(x)  # Shape: [batch, seq_len, hidden_size]

        # Get final timestep and apply dropout
        out = self.dropout(out[:, -1, :])

        # Final classification layer
        # return self.fc(out)
        return out


class Stackxlstm(nn.Module):
    def __init__(self, modellist):
        super().__init__()
        self.audio_model_speaker = modellist[0]
        self.audio_model_con = modellist[1]
        self.text_model_speaker = modellist[2]
        self.text_model_con = modellist[3]

    def forward(self, a1, t1, a2, t2):
        a1 = self.audio_model_speaker(a1)
        t1 = self.text_model_speaker(t1)
        a2 = self.audio_model_con(a2)
        t2 = self.text_model_con(t2)

        return a1, t1, a2, t2


class ForgetGateStackedxlstm(nn.Module):
    def __init__(self, num_features, model, hidden_size, num_classes):
        super().__init__()
        largest = torch.rand(4) * 1 + 2  # Larger values around 3
        middle = torch.rand(4) * 1  # Around 0 (moderate)
        smallest = torch.rand(4) * 1 - 2  # Smaller values around -3

        # Concatenate them into a single tensor
        # init_values = torch.cat([largest, middle, smallest])

        # Register as a learnable parameter
        self.forget_gate = nn.Linear(512, 11)  # 786 is the x
        # self.forget_gate = nn.Parameter(torch.randn(num_features))  # Learnable forget weights
        self.sigmoid = nn.Sigmoid()
        self.stackmodel = model
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: List of 12 feature matrices, each of shape (batch_size, seq_len, dim, 12) or similar.
        Returns: List of gated feature matrices.
        """
        tensors = []
        j = 0
        current_state = x[..., 0].squeeze(-1)
        for i in range(0, 12, 4):
            a1 = x[..., i].squeeze(-1)
            a2 = x[..., i + 1].squeeze(-1)
            t1 = x[..., i + 2].squeeze(-1)
            t2 = x[..., i + 3].squeeze(-1)
            result = self.stackmodel[j](a1, t1, a2, t2)
            tensors.append(result[0])
            tensors.append(result[1])
            tensors.append(result[2])
            tensors.append(result[3])

            j += 1
        tensors = torch.stack(tensors, dim=0)
        batch = tensors.shape[1]
        gate_values = torch.mean(self.sigmoid(self.forget_gate(current_state.reshape(batch, -1))),
                                 axis=0)  # (11,) shape, one value per feature
        gated_features = [tensors[0]]
        gated_features.extend([tensors[i + 1] * gate_values[i] for i in range(len(tensors) - 1)])  # Feature-wise gating
        out = torch.stack(gated_features, dim=0).permute(1, 0, 2).reshape(batch, -1)
        # out1 = tensors.permute(1, 0, 2).reshape(batch, -1)
        return self.fc(out)


# Lets Load audio features
audio_data = np.load("../clap_with_oppo_IEMOCAP_audio_features_512.npy", allow_pickle=True).item()
audio_features = audio_data['features']

# Lets Load text features
text_data = np.load("../clap_with_oppo_IEMOCAP_text_features_512.npy", allow_pickle=True).item()
text_features = text_data['features']
text_labels = text_data['labels']
fileids = np.array(audio_data["fileid"])

metadata = pd.read_csv('../IEMOCAP_extracted_emotions.csv')

file_to_prev = dict(zip(metadata['fileid'], metadata['prev']))

# We just propose t-2 ~ t
a1_0, a2_0, t1_0, t2_0 = [], [], [], []
a1_1, a2_1, t1_1, t2_1 = [], [], [], []
a1_2, a2_2, t1_2, t2_2 = [], [], [], []
y = []
# Define the selected classes: angry (0), happy (5) and excitmenr (2), neutral (6), sad (8)
selected_classes = [0, 5, 6, 8]
label_mapping = {0: 0, 5:1, 6: 2, 8: 3 }
# selected_classes = [0,2, 5, 6, 8]
# label_mapping = {0: 0, 2:1, 5:1, 6: 2, 8: 3 }
zeros_array = np.zeros(512)


def fill_features(a1, a2, t1, t2, pos):
    a1.append(audio_data['features'][pos])
    a2.append(audio_data['oppo_prev'][pos])
    t1.append(text_data['features'][pos])
    t2.append(text_data['oppo_prev'][pos])
    return a1, a2, t1, t2


def fill_zero(a1, a2, t1, t2):
    a1.append(zeros_array)
    a2.append(zeros_array)
    t1.append(zeros_array)
    t2.append(zeros_array)
    return a1, a2, t1, t2


# organise and clean the data
for pos in range(len(audio_data['fileid'])):
    if audio_data['labels'][pos] not in selected_classes:
        continue
    a1_0, a2_0, t1_0, t2_0 = fill_features(a1_0, a2_0, t1_0, t2_0, pos)
    y.append(text_data['labels'][pos])

    prev = file_to_prev[audio_data['fileid'][pos]]

    try:
        pos1 = np.where(audio_data['fileid'] == prev)[0][0]
        a1_1, a2_1, t1_1, t2_1 = fill_features(a1_1, a2_1, t1_1, t2_1, pos1)

    except:
        a1_1, a2_1, t1_1, t2_1 = fill_zero(a1_1, a2_1, t1_1, t2_1)
        a1_2, a2_2, t1_2, t2_2 = fill_zero(a1_2, a2_2, t1_2, t2_2)
        continue

    try:
        prev = file_to_prev[audio_data['fileid'][pos1]]
        pos2 = np.where(audio_data['fileid'] == prev)[0][0]
        a1_2, a2_2, t1_2, t2_2 = fill_features(a1_2, a2_2, t1_2, t2_2, pos2)
    except:
        a1_2, a2_2, t1_2, t2_2 = fill_zero(a1_2, a2_2, t1_2, t2_2)

a1_0, a2_0, t1_0, t2_0 = np.vstack(a1_0), np.vstack(a2_0), np.vstack(t1_0), np.vstack(t2_0),
a1_1, a2_1, t1_1, t2_1 = np.vstack(a1_1), np.vstack(a2_1), np.vstack(t1_1), np.vstack(t2_1),
a1_2, a2_2, t1_2, t2_2 = np.vstack(a1_2), np.vstack(a2_2), np.vstack(t1_2), np.vstack(t2_2),

# y = np.array([label_mapping[label] for label in y])

filtered_fileids = []
filtered_labels = []

# 确保 `fileid` 和 `y` 对应
for fid, label in zip(audio_data["fileid"], audio_data["labels"]):
    if label in selected_classes:
        filtered_fileids.append(fid)
        filtered_labels.append(label_mapping[label])

#
fileids_filtered = np.array(filtered_fileids)
y = np.array(filtered_labels)
from collections import Counter
# 统计每种标签的数量
label_counts = Counter(y)
total_labels = len(y)

# 计算并打印每个标签的数量及占比
print(f"Total Labels: {total_labels}")
for label, count in label_counts.items():
    percentage = (count / total_labels) * 100
    print(f"Label {label}: {count} ({percentage:.2f}%)")

features = np.stack([a1_0, a2_0, t1_0, t2_0, a1_1, a2_1, t1_1, t2_1, a1_2, a2_2, t1_2, t2_2], axis=2)

time_steps = 16
features_per_step = features.shape[1] // time_steps
features = features.reshape(features.shape[0], time_steps, features_per_step, 12)

# Split into train (80%), validation (10%), test (10%)
train_data, temp_data, train_labels, temp_labels = train_test_split(features, y, test_size=0.2,
                                                                    random_state=42)  # shape [number of sample, hidden dim, 12], 12 as we have t-2 ~ t and each has 4
valid_data, test_data, valid_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5,
                                                                    random_state=42)
train_data, temp_data, train_labels, temp_labels, train_fileids, temp_fileids = train_test_split(
    features, y, fileids_filtered, test_size=0.2, stratify=y, random_state=42
)

valid_data, test_data, valid_labels, test_labels, valid_fileids, test_fileids = train_test_split(
    temp_data, temp_labels, temp_fileids, test_size=0.5, stratify=temp_labels,  random_state=42
)


X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_val = torch.tensor(valid_data, dtype=torch.float32)
y_val = torch.tensor(valid_labels, dtype=torch.long)
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)
# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dialog_dict = defaultdict(list)

for idx, fileid in enumerate(fileids_filtered):
    dialog_id = re.sub(r'_F\d+$|_M\d+$', '', fileid)  # 去掉性别标识，获取对话 ID
    dialog_dict[dialog_id].append(idx)

# 获取所有对话 ID
dialog_ids = list(dialog_dict.keys())

# 按对话 ID 进行划分
train_dialogs, temp_dialogs = train_test_split(dialog_ids, test_size=0.2, random_state=42)
valid_dialogs, test_dialogs = train_test_split(temp_dialogs, test_size=0.5, random_state=42)

# 获取每个数据集对应的 utterance 索引
train_idx = [idx for dialog in train_dialogs for idx in dialog_dict[dialog]]
valid_idx = [idx for dialog in valid_dialogs for idx in dialog_dict[dialog]]
test_idx = [idx for dialog in test_dialogs for idx in dialog_dict[dialog]]

# 根据索引获取数据
train_data, train_labels = features[train_idx], y[train_idx]
valid_data, valid_labels = features[valid_idx], y[valid_idx]
test_data, test_labels = features[test_idx], y[test_idx]

train_fileids = fileids_filtered[train_idx]
valid_fileids = fileids_filtered[valid_idx]
test_fileids = fileids_filtered[test_idx]

# 转换成 PyTorch Tensor
X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_val = torch.tensor(valid_data, dtype=torch.float32)
y_val = torch.tensor(valid_labels, dtype=torch.long)
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"train dialog number: {len(train_dialogs)}")
print(f"val dialog number: {len(valid_dialogs)}")
print(f"test dialog number: {len(test_dialogs)}")

print(f"train Utterance num: {len(train_idx)}")
print(f"val Utterance num: {len(valid_idx)}")
print(f"test Utterance num: {len(test_idx)}")

from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_size = features_per_step  # 32 based on your 512/16 split
hidden_size = 512
num_layers = 8
#
model_list = []

for i in range(12):
    model = Onexlstm(input_size, hidden_size, num_layers).to(device)
    model_list.append(model)

stack_model1 = Stackxlstm(model_list[0:4]).to(device)
stack_model2 = Stackxlstm(model_list[4:8]).to(device)
stack_model3 = Stackxlstm(model_list[8:12]).to(device)

final_model = ForgetGateStackedxlstm(12, [stack_model1, stack_model2, stack_model3], hidden_size * 12, 4).to(device)

# Try concat at first place
# final_model = Onexlstm(input_size * 12, hidden_size, num_layers, 4).to(device)

optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()

best_val_f1 = 0.0
early_stop = 20
early = 0
for epoch in range(100):
    # Training phase
    final_model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    # Validation phase
    final_model.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    # Calculate average losses and accuracy
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_true, val_preds)

    # Calculate F1 score (using 'macro' average; change as needed)
    val_f1 = f1_score(val_true, val_preds, average='macro')

    # Scheduler step (using validation loss in this example)
    # scheduler.step(val_loss)

    print(f"Epoch {epoch + 1:03d}: "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f}")

    # Save the best model based on validation F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(final_model, './best_model_2.pt')
        print(f"--> New best model saved with Val f1: {best_val_f1:.4f}")
        early = 0
    else:
        early += 1
    if early >= early_stop:
        break

warnings.filterwarnings("ignore", category=FutureWarning)
# Lets map the labels
target_mapping = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

target_names = [target_mapping[i] for i in sorted(target_mapping.keys())]

# --------------------------------------------------
# Final Evaluation

test_model = torch.load('try.pt', map_location=device)
test_model.eval()  # Set to evaluation mode if needed
test_preds = []
test_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = test_model(inputs)
        # print('input,', inputs[0][0])
        # print(('output,',outputs[0]))

        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

accuracy = accuracy_score(test_true, test_preds,)
wf1 = f1_score(test_true, test_preds, average='weighted')
print("\nFinal Test Results on iemocap data:")
print(f"Accuracy: {accuracy:.4f}")
print(f"weight-F1: {wf1:.4f}")
report = classification_report(test_true, test_preds, target_names=target_names, digits=4, zero_division=0)
print("\nClassification Report:")
print(report)

out_dict = {}  # 存储 {fileid: logits}
emo_dict = {}  # 存储 {fileid: true_label}
dialogs = defaultdict(list)  # {dialog_id: [fileid1, fileid2, ...]}

# 取 fileids_test（对应 test_loader 顺序）
fileid_list = test_fileids.tolist()


with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = test_model(inputs)  # 获取 logits
        logits = outputs.cpu().numpy()

        for i in range(len(inputs)):  # 遍历当前 batch 内样本
            fileid = fileid_list[batch_idx * test_loader.batch_size + i]  # 获取 fileid
            out_dict[fileid] = logits[i]  # 存储 logits 结果
            emo_dict[fileid] = labels[i].item()  # 存入真实情感标签


            dialog_id = re.sub(r'_F\d+$|_M\d+$', '', fileid)
            dialogs[dialog_id].append(fileid)

print(f"✅ 生成 out_dict: {len(out_dict)} 条预测")
print(f"✅ 生成 emo_dict: {len(emo_dict)} 条真实标签")
print(f"✅ 生成 dialogs: {len(dialogs)} 个对话")
print('dialogs 长这样', dialogs)

sample_fid = next(iter(out_dict))
print(f"\n示例 fileid: {sample_fid}")
print(f"  logits: {out_dict[sample_fid]}")
print(f"  预测类别: {np.argmax(out_dict[sample_fid])}")
print(f"  真实类别: {emo_dict[sample_fid]}")

# print("Forget Gate Weights:\n", final_model.forget_gate.data)  # or .detach()

print(emo_dict, out_dict)


# # 生成 dialogs
# dialogs = defaultdict(list)
# for fileid in metadata["fileid"][:len(y_val)]:
#     dialog_id = re.sub(r'_F\d+$|_M\d+$', '', fileid)
#     dialogs[dialog_id].append(fileid)
# print('len_val', len(y_val))
# # -------------
# # 用全部的数据集
# # with torch.no_grad():
# #     for dataset, y_dataset in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
# #         for idx in range(len(y_dataset)):
# #             fileid = metadata["fileid"][idx]
# #             inputs = dataset[idx].unsqueeze(0).to(device)
# #
# #             logits = final_model(inputs).squeeze().cpu().numpy()
# #             out_dict[fileid] = logits
# #
# #             emo_dict[fileid] = y_dataset[idx].item()
# #
# # dialogs = defaultdict(list)
# # for fileid in out_dict.keys():
# #     dialog_id = re.sub(r'_F\d+$|_M\d+$', '', fileid)
# #     dialogs[dialog_id].append(fileid)
# #


spk_dialogs = utils.split_dialog(dialogs)
#
# # print(emo_dict)
# # print('-------')
# # print(out_dict)
# # print('--------')
# # print(dialogs)
#
pred_classes = [np.argmax(v) for v in out_dict.values()]
unique_pred = np.unique(pred_classes)
print(f"预测类别分布: {np.bincount(pred_classes)}")
print(f"唯一预测类别: {unique_pred}")

nan_count = sum(np.isnan(v).any() for v in out_dict.values())
print(f"包含NaN的预测数: {nan_count}")

sample_fid = next(iter(out_dict))
print(f"示例预测 '{sample_fid}':")
print(f"  logits: {out_dict[sample_fid]}")
print(f"  预测类别: {np.argmax(out_dict[sample_fid])}")
print(f"  真实标签: {emo_dict[sample_fid]}")

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO,
                    datefmt='%I:%M:%S')

args = parse_args()

print("\n=== 对话结构验证 ===")
empty_dialogs = [k for k, v in dialogs.items() if not v]
print(f"empty dialog: {len(empty_dialogs)}")

# 检查每个对话中的utterance是否都在emo_dict中
invalid_utterances = []
for dialog_id, utterances in dialogs.items():
    for utt in utterances:
        if utt not in emo_dict:
            invalid_utterances.append(utt)

print(f"error utterance in dialog: {len(invalid_utterances)}")
if invalid_utterances:
    print("example error utterance:", invalid_utterances[:3])
print('=' * 60 + '\n')
logging.info('Parameters are:\n%s\n', json.dumps(vars(args), sort_keys=False, indent=4))
print('=' * 60 + '\n')

if args.transition_bias > 0:
    # Use given p_0
    p_0 = args.transition_bias

else:
    # Estimate p_0 of ALL dialogs.
    p_0, total_transit = utils.transition_bias(spk_dialogs, emo_dict)

    print("\n" + "#" * 50)
    logging.info('p_0: %.3f , total transition: %d\n' % (p_0, total_transit))
    print("#" * 50)

    bias_dict = utils.get_val_bias(spk_dialogs, emo_dict)
    print("#" * 50 + "\n")

trace = []
label = []
org_pred = []
DED = bs.BeamSearch(p_0, args.crp_alpha, args.num_state,
                    args.beam_size, args.test_iteration, emo_dict, out_dict)

for i, dia in enumerate(dialogs):
    logging.info("Decoding dialog: {}/{}, {}".format(i, len(dialogs), dia))

    # Apply p_0 estimated from other 4 sessions.
    DED.transition_bias = bias_dict[dia[:5]]

    # Beam search decoder
    out = DED.decode(dialogs[dia])

    trace += out
    label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]
    org_pred += [np.argmax(out_dict[utt]) for utt in dialogs[dia]]
    if args.verbosity > 0:
        logging.info("Output: {}\n".format(out))
print("#" * 50 + "\n")

# Print the results of emotino classifier module
per_class_acc, per_class_f1, w_acc, w_f1, uar, cm, report = utils.evaluate(org_pred, label)
logging.info('Original performance:')
print("Per-Class Accuracy:", per_class_acc)
print("Per-Class F1-score:", per_class_f1)
print(f"Weighted Accuracy: {w_acc:.4f}")
print(f"Weighted F1-score: {w_f1:.4f}")
print(f"UAR (Unweighted Average Recall): {uar:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Eval ded outputs
results = vars(args)
per_class_acc, per_class_f1, w_acc, w_f1, uar, cm, report = utils.evaluate(trace, label)
logging.info('DED performance: ')
print("Per-Class Accuracy:", per_class_acc)
print("Per-Class F1-score:", per_class_f1)
print(f"Weighted Accuracy: {w_acc:.4f}")
print(f"Weighted F1-score: {w_f1:.4f}")
print(f"UAR (Unweighted Average Recall): {uar:.4f}")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Save results
# results['uar'] = uar
# results['acc'] = acc
# results['conf'] = str(conf)
# results['wf1'] = acc
# logging.info('Save results:')
# logging.info('\n%s\n', json.dumps(results, sort_keys=False, indent=4))
# json.dump(results, open(args.out_dir + '/%s.json' % args.result_file, "w"))

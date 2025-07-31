import numpy as np
from sklearn.metrics import confusion_matrix,classification_report, recall_score, accuracy_score, f1_score


def split_dialog(dialogs):
    """Split utterances in a dialog into separate speakers' utterances.

    Args:
        dialogs: dict, mapping of dialog_id -> list of utterances
                 Example: {2: ['27_Ross', '28_Rachel', '29_Ross', '30_Ross', '31_Rachel', '32_Ross']}

    Returns:
        spk_dialogs: dict, mapping of dialog_id_speaker -> list of utterances
                     Example: {2_Ross: ['27_Ross', '29_Ross', '30_Ross', '32_Ross'],
                               2_Rachel: ['28_Rachel', '31_Rachel']}
    """
    spk_dialogs = {}

    for dialog_id, utt_list in dialogs.items():
        for utt_id in utt_list:
            # 从 fileid_Speaker 解析出 speaker
            parts = utt_id.split('_')
            if len(parts) < 2:
                print(f"Warning: Unexpected format in {utt_id}")
                continue

            fileid, speaker = parts[0], '_'.join(parts[1:])  # 提取 fileid 和 speaker

            # 生成唯一的对话-说话人 ID，例如 "2_Ross"
            spk_dialog_id = f"{dialog_id}_{speaker}"

            # 初始化列表并存入 utterance
            if spk_dialog_id not in spk_dialogs:
                spk_dialogs[spk_dialog_id] = []
            spk_dialogs[spk_dialog_id].append(utt_id)

    return spk_dialogs


def transition_bias(spk_dialogs, emo, val=None):
    """Estimate the transition bias of emotion. See eq (5) in the paper.

    Args:
        spk_dialogs: dict, {dialog_id_speaker: [utterance1, utterance2, ...]}
        emo: dict, {utterance_id: emotion_label}
        val: str, session to exclude.

    Returns:
        bias: p_0 in eq (4).
    """
    transit_num = 0
    total_transit = 0

    for spk_dialog_id, utt_list in spk_dialogs.items():
        # 过滤掉特定的 `session`
        if val and str(spk_dialog_id).startswith(val):
            continue

        for i in range(len(utt_list) - 1):
            if utt_list[i] in emo and utt_list[i + 1] in emo:
                # 计算情感转换次数
                transit_num += (emo[utt_list[i]] != emo[utt_list[i + 1]])
            total_transit += 1  # 计算总的转换次数

    if total_transit == 0:
        return 0, 0  # 防止除零错误

    bias = (transit_num + 1) / total_transit  # 平滑处理
    return bias, total_transit


# def get_val_bias(dialogs, emo_dict):
#
#     dialog_ids = list(dialogs.keys())  # 获取所有对话 ID
#     bias_dict = {}
#
#     for i in range(len(dialog_ids)):
#         val = dialog_ids[i]  # 当前对话 ID 作为验证集
#         train_dialogs = {k: v for k, v in dialogs.items() if k != val}  # 训练集 = 其余对话
#
#         # 计算去掉该对话后的 transition bias
#         p_0, _ = transition_bias(train_dialogs, emo_dict)
#
#         print(f"Transition bias without {val}: {p_0:.3f}")
#         bias_dict[val] = p_0  # 存入 bias 结果
#     return bias_dict
def get_val_bias(dialogs, emo_dict):
    """
    计算去掉某个 `speaker` 后的 transition bias。

    Args:
        dialogs: dict, {dialog_id_speaker: [utterance1, utterance2, ...]}
        emo_dict: dict, {utterance_id: emotion_label}

    Returns:
        bias_dict: dict, {speaker: transition_bias}
    """

    # 1️⃣ 解析 `dialog_id_speaker`，按 `speaker` 分组
    speaker_to_utterances = {}  # {speaker: [utterance1, utterance2, ...]}

    for dialog_id_speaker, utt_list in dialogs.items():
        speaker = dialog_id_speaker.split('_')[-1]  # 提取 speaker
        if speaker not in speaker_to_utterances:
            speaker_to_utterances[speaker] = []
        speaker_to_utterances[speaker].extend(utt_list)  # 收集 speaker 说的所有 utterances

    # 2️⃣ 遍历 `speaker`，计算去掉该 speaker 后的 transition bias
    bias_dict = {}

    for speaker in speaker_to_utterances.keys():
        # 过滤掉当前 speaker 说的所有 utterances
        train_dialogs = {
            k: [utt for utt in v if utt not in speaker_to_utterances[speaker]]
            for k, v in dialogs.items()
        }
        train_dialogs = {k: v for k, v in train_dialogs.items() if v}  # 移除空对话

        # 计算 transition bias
        p_0, _ = transition_bias(train_dialogs, emo_dict)

        print(f"Transition bias without {speaker}: {p_0:.3f}")
        bias_dict[speaker] = p_0  # 存入 bias 结果

    return bias_dict

# def find_last_idx(trace_speakers, speaker):
#   """Find the index of speaker's last utterance."""
#   for i in range(len(trace_speakers)):
#     if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
#         return len(trace_speakers) - (i+1)
def find_last_idx(trace_speakers, speaker):
    """Find the index of speaker's last utterance in the sequence."""
    for i in range(len(trace_speakers) - 1, -1, -1):  # 从最后一个元素往前找
        last_speaker = trace_speakers[i].split('_')[-1]  # 提取 Speaker 名字
        if last_speaker == speaker:
            return i  # 返回找到的索引
    return None  # 如果没有找到，返回 None



def cross_entropy(targets, predictions, epsilon=1e-12):
  """Computes cross entropy between targets (one-hot) and predictions. 
  Args: 
    targets: (1, num_state) ndarray.   
    predictions: (1, num_state) ndarray.

  Returns: 
    cross entropy loss.
  """
  targets = np.array(targets)
  predictions = np.array(predictions)
  ce = -np.sum(targets*predictions)
  return ce

def convert_to_index(emotion):
  """convert emotion to index """
  map_emo = { 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
  if emotion in map_emo.keys():
    return map_emo[emotion]
  else:
    return -1


def evaluate(trace, label):
    # 只评估属于定义的 4 种情绪状态的语句
    label, trace = np.array(label), np.array(trace)
    index = label != -1
    label, trace = label[index], trace[index]

    # 类别映射
    target_mapping = {0: "anger", 1: "disgust", 2: "fear", 3: "joy",
                  4: "neutral", 5: "sadness", 6: "surprise"}
    target_names = [target_mapping[i] for i in sorted(target_mapping.keys())]

    # 计算分类报告
    report_dict = classification_report(label, trace, target_names=target_names, digits=4, zero_division=0, output_dict=True)

    # 计算每个类别的 Accuracy（Per-Class Accuracy = Recall）
    per_class_accuracy = {cls: report_dict[cls]["recall"] for cls in target_names}
    per_class_f1 = {cls: report_dict[cls]["f1-score"] for cls in target_names}

    # 计算 Weighted Accuracy（加权准确率）
    total_samples = sum(report_dict[cls]["support"] for cls in target_names)
    weighted_accuracy = sum(
        (report_dict[cls]["support"] / total_samples) * report_dict[cls]["recall"]
        for cls in target_names
    )

    # 计算 Weighted F1-score
    weighted_f1 = f1_score(label, trace, average="weighted")

    # 计算 UAR（Unweighted Average Recall，即 Macro Recall）
    uar = recall_score(label, trace, average="macro")

    # 计算混淆矩阵
    cm = confusion_matrix(label, trace)

    return per_class_accuracy, per_class_f1, weighted_accuracy, weighted_f1, uar, cm, classification_report(label, trace, target_names=target_names, digits=4, zero_division=0)


if __name__ == '__main__':
    dialogs = {
        2: ['27_Ross', '28_Rachel', '29_Ross', '30_Ross', '31_Rachel', '32_Ross','33_Bob','34_Bob'],
    }

    emo = {
        '27_Ross': 0, '28_Rachel': 1,
        '29_Ross': 0, '30_Ross': 0,
        '31_Rachel': 0, '32_Ross': 0,
        '33_Bob': 0,'34_Bob': 0
    }

    # 拆分对话为不同说话人的子对话
    spk_dialogs = split_dialog(dialogs)

    # 计算情感转换偏差
    bias, total_transit = transition_bias(spk_dialogs, emo)

    print(f"Split Dialogs: {spk_dialogs}")
    print(f"Transition Bias: {bias:.4f}, Total Transitions: {total_transit}")

    trace_speakers = ['18_Ross', '22_Rachel', '27_Ross', '28_Rachel', '29_Ross']
    speaker = "Ross"

    last_idx = find_last_idx(trace_speakers, speaker)
    print(f"Last index for {speaker}: {last_idx}")

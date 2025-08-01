import numpy as np
from sklearn.metrics import confusion_matrix,classification_report, recall_score, accuracy_score, f1_score

def split_dialog(dialogs):
  """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
     See eq (5) in the paper.
  Arg:
    dialogs: dict, for example, utterances of two speakers in dialog_01: 
            {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
  Return:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
            {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
             dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
  """

  spk_dialogs = {}
  for dialog_id in dialogs.keys():
    spk_dialogs[dialog_id+'_M'] = []
    spk_dialogs[dialog_id+'_F'] = []
    for utt_id in dialogs[dialog_id]:
      if utt_id[-4] == 'M':
        spk_dialogs[dialog_id+'_M'].append(utt_id)
      elif utt_id[-4] == 'F':
        spk_dialogs[dialog_id+'_F'].append(utt_id)

  return spk_dialogs

def transition_bias(spk_dialogs, emo, val=None):
  """Estimate the transition bias of emotion. See eq (5) in the paper.
  Args:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
    emo: dict, map from utt_id to emotion state.
    val: str, validation session. If given, calculate the trainsition bias except 
         dialogs in the validation session. For example, 'Ses01'.

  Returns: 
    bias: p_0 in eq (4).
  """
  transit_num = 0
  total_transit = 0
  count = 0
  num = 0
  for dialog_id in spk_dialogs.values():
    # if val and val == dialog_id[0][:5]:
    if val and val == dialog_id[:5]:
        print('aaa')
        continue

    for entry in range(len(dialog_id) - 1):
      transit_num += (emo[dialog_id[entry]] != emo[dialog_id[entry + 1]])
    total_transit += (len(dialog_id) - 1)

  bias = (transit_num + 1) / total_transit

  return bias, total_transit

def get_val_bias(dialog, emo_dict):
    """Get p_0 estimated from training sessions."""

    session = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    bias_dict = {}
    for i in range(len(session)):
      val = session[i]
      train_sessions = session[:i] + session[i+1:]
      p_0, _ = transition_bias(dialog, emo_dict, val)
      print("Transition bias of { %s }: %.3f" % (' ,'.join(train_sessions), p_0))
      bias_dict[val] = p_0

    return bias_dict

def find_last_idx(trace_speakers, speaker):
  """Find the index of speaker's last utterance."""
  for i in range(len(trace_speakers)):
    if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
        return len(trace_speakers) - (i+1)

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
  map_emo = { 0:0, 1:1, 2:2, 3:3}
  if emotion in map_emo.keys():
    return map_emo[emotion]
  else:
    return -1

# def evaluate(trace, label):
#   # Only evaluate utterances labeled in defined 4 emotion states
#   label, trace = np.array(label), np.array(trace)
#   index = label != -1
#   label, trace = label[index], trace[index]
#   target_mapping = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
#   target_names = [target_mapping[i] for i in sorted(target_mapping.keys())]
#   report = classification_report(trace, label, target_names=target_names, digits=4, zero_division=0)
#   return (recall_score(label, trace, average='macro'),
#           accuracy_score(label, trace),
#           confusion_matrix(label, trace),
#           f1_score(label, trace, average='weighted'),
#           report
#           )
def evaluate(trace, label):
    # 只评估属于定义的 4 种情绪状态的语句
    label, trace = np.array(label), np.array(trace)
    index = label != -1
    label, trace = label[index], trace[index]

    # 类别映射
    target_mapping = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
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
    dialog = {'Ses05M_script03_2_M': ['Ses05M_script03_2_M042', 'Ses05M_script03_2_M043',
                'Ses05M_script03_2_M044', 'Ses05M_script03_2_M045']}
    emo = {'Ses05M_script03_2_M042': 'ang', 'Ses05M_script03_2_M043': 'ang',
                'Ses05M_script03_2_M044': 'ang', 'Ses05M_script03_2_M045': 'ang'}

    spk_dialog = split_dialog(dialog)
    bias, total_transit = transition_bias(spk_dialog, emo)
    crp_alpha = 1
    print('Transition bias: {} , Total transition: {}'.format(bias, total_transit))


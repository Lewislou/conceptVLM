import pandas as pd
import json
import collections
import pandas as pd
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
def calculate_exactmatch(pred, gt):
    return 1 if pred.strip() == gt.strip() else 0
nltk.download('punkt_tab')
def calculate_f1score(pred, gt):
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall
# 读取 JSON 文件
with open('internvl_v2_8B_val_v5_fulldataset_closed.json', 'r') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 过滤出 answer_type 为 "CLOSE" 的数据
df_close = df[df['answer_type'] == "CLOSE"]

# 计算模态准确率
# CFP 模态
cfp_gt = df_close[df_close['gt_ans'] == "This is a color fundus image."]
cfp_count_gt = cfp_gt.shape[0]
cfp_count_answer = cfp_gt['answer'].str.contains("Color Fundus|color fundus|CF|Color", case=False).sum()

# FFA 模态
ffa_gt = df_close[df_close['gt_ans'] == "This is a fundus fluorescein angiography (FFA) image."]
ffa_count_gt = ffa_gt.shape[0]
ffa_count_answer = ffa_gt['answer'].str.contains("Fundus Fluorescein Angiography|fluorescein angiography|FFA|Angiography|Fluorescein", case=False).sum()

# OCT 模态
oct_gt = df_close[df_close['gt_ans'] == "This is an optical coherence tomography (OCT) image."]
oct_count_gt = oct_gt.shape[0]
oct_count_answer = oct_gt['answer'].str.contains("Optical Coherence Tomography|optical coherence tomography|OCT|Optical", case=False).sum()

# 计算模态准确率
cfp_accuracy = cfp_count_answer / cfp_count_gt if cfp_count_gt > 0 else 0
ffa_accuracy = ffa_count_answer / ffa_count_gt if ffa_count_gt > 0 else 0
oct_accuracy = oct_count_answer / oct_count_gt if oct_count_gt > 0 else 0

average_modality_accuracy = (cfp_count_answer + ffa_count_answer + oct_count_answer) / (cfp_count_gt + ffa_count_gt + oct_count_gt) if (cfp_count_gt + ffa_count_gt + oct_count_gt) > 0 else 0

# 输出模态准确率
print(f"CFP 模态准确率: {cfp_accuracy:.2%}")
print(f"FFA 模态准确率: {ffa_accuracy:.2%}")
print(f"OCT 模态准确率: {oct_accuracy:.2%}")
print(f"平均模态准确率: {average_modality_accuracy:.2%}")

# 计算眼别准确率
left_eye_gt = df_close[df_close['gt_ans'] == "Left eye."]
left_eye_count_gt = left_eye_gt.shape[0]
left_eye_count_answer = left_eye_gt['answer'].str.contains("left eye|left|Left", case=False).sum()

right_eye_gt = df_close[df_close['gt_ans'] == "Right eye."]
right_eye_count_gt = right_eye_gt.shape[0]
right_eye_count_answer = right_eye_gt['answer'].str.contains("right eye|right|Right", case=False).sum()

left_eye_accuracy = left_eye_count_answer / left_eye_count_gt if left_eye_count_gt > 0 else 0
right_eye_accuracy = right_eye_count_answer / right_eye_count_gt if right_eye_count_gt > 0 else 0

average_eye_accuracy = (left_eye_count_answer + right_eye_count_answer) / (left_eye_count_gt + right_eye_count_gt) if (left_eye_count_gt + right_eye_count_gt) > 0 else 0

# 输出眼别准确率
print(f"左眼准确率: {left_eye_accuracy:.2%}")
print(f"右眼准确率: {right_eye_accuracy:.2%}")
print(f"平均眼别准确率: {average_eye_accuracy:.2%}")

# 提取诊断名称
df_close['diagnosis'] = df_close['gt_ans']  # .str.extract(r'The possible diagnosis of this image is (.+?)\.')[0]
df_diagnosis = df_close[df_close['question'].str.contains("diagnosis", case=False, na=False)]
# 计算诊断准确率
correct_diagnosis_count = 0
diagnosis_count = df_diagnosis['diagnosis'].notna().sum()
diag_dict = {}

# 遍历每一行，检查 answer 中是否包含对应的诊断
for index, row in df_diagnosis.iterrows():
    if pd.notna(row['diagnosis']):
        # 检查预测的答案是否与诊断匹配
        if row['answer'].split('\n')[0] in row['diagnosis']:
            correct_diagnosis_count += 1
        else:
            # 如果不匹配，记录下来
            if row['diagnosis'] in diag_dict.keys():
                diag_dict[row['diagnosis']].append(row['answer'])
            else:
                diag_dict[row['diagnosis']] = [row['answer']]
data = [{'Key': key, 'Value': value, 'length': len(value)} for key, value in diag_dict.items()]
      
diagnosis_accuracy = correct_diagnosis_count / diagnosis_count if diagnosis_count > 0 else 0
df_diag = pd.DataFrame(data)

# 指定输出的Excel文件路径
#excel_file_path = 'diag_dict.xlsx'

# 将DataFrame写入Excel文件
#df_diag.to_excel(excel_file_path, index=False)

#print(f"Data has been saved to {excel_file_path}")
# 输出诊断准确率
print(f"诊断准确率: {diagnosis_accuracy:.2%}")

smoother = SmoothingFunction()
exact_scores = collections.defaultdict(list)
f1_scores = collections.defaultdict(list)
bleu_scores = collections.defaultdict(list)
df_close = df[df['answer_type'] == "OPEN"]

# 确保 df_close 是 DataFrame 类型
if isinstance(df_close, pd.DataFrame):
    for index, row in df_close.iterrows():
        gt_value = row['gt_ans'].lower()
        pred_value = row['answer'].lower()
        question_id = row['question_id']
        question = row['question']  # 获取问题文本以检查是否包含 'modality'

        #if 'modality' in question.lower():  # 确保问题是关于模态的
        exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
        exact_scores['q_id'].append(question_id)

        f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
        f1_scores['f1'].append(f1_score)
        f1_scores['precision'].append(precision)
        f1_scores['recall'].append(recall)
        f1_scores['q_id'].append(question_id)

        # 计算BLEU-1和BLEU-4
        reference = [word_tokenize(gt_value)]
        candidate = word_tokenize(pred_value)

        bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
        bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method4)

        bleu_scores['bleu1'].append(bleu1)
        bleu_scores['bleu4'].append(bleu4)
        bleu_scores['q_id'].append(question_id)

    # 计算平均分
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit']) if exact_scores['hit'] else 0.0
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1']) if f1_scores['f1'] else 0.0
    precision = sum(f1_scores['precision']) / len(f1_scores['precision']) if f1_scores['precision'] else 0.0
    recall = sum(f1_scores['recall']) / len(f1_scores['recall']) if f1_scores['recall'] else 0.0    
    bleu1_score = sum(bleu_scores['bleu1']) / len(bleu_scores['bleu1']) if bleu_scores['bleu1'] else 0.0
    bleu4_score = sum(bleu_scores['bleu4']) / len(bleu_scores['bleu4']) if bleu_scores['bleu4'] else 0.0

    print(tabulate(
        [
            ['Exact Match Score', exact_score * 100],
            ['F1 Score', f1_score * 100],
            ['Precision', precision * 100],
            ['Recall', recall * 100],
            ['BLEU-1 Score', bleu1_score * 100],
            ['BLEU-4 Score', bleu4_score * 100],
        ],
        headers=['Metric', 'Performance']
    ))
else:
    print("df_close is not a DataFrame.")
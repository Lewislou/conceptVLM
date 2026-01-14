import pandas as pd
import json
import collections
import pandas as pd
from tabulate import tabulate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
import nltk

import re

# 假设你已经有calculate_exactmatch, calculate_f1score, align_dict_lists函数定义

def contains_chinese(text):
    """Check if the text contains any Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def filter_english_questions(data_list):
    """Filter out items with questions containing Chinese characters."""
    return [item for item in data_list if not contains_chinese(item.get('question', ''))]
    
def calculate_metrics_for_dataset(dataset_items):
    scores = collections.defaultdict(list)
    closed_scores = collections.defaultdict(list)
    closed_questions_count = 0
    closed_questions_correct = 0
    
    for item in dataset_items:
        gt_value = item.get('gt_ans', '').lower()
        pred_value = item.get('answer', '').lower()
        answer_type = item.get('answer_type', 'OPEN').upper()

        if answer_type == 'OPEN':
            scores['exact_match'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            scores['f1'].append(f1_score)
            scores['precision'].append(precision)
            scores['recall'].append(recall)
        elif answer_type == 'CLOSE':
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            closed_scores['f1'].append(f1_score)
            closed_scores['precision'].append(precision)
            closed_scores['recall'].append(recall)
            closed_questions_count += 1
            if (gt_value in pred_value) or (pred_value in gt_value):
                closed_questions_correct += 1

    return scores, closed_scores, closed_questions_count, closed_questions_correct
    
def group_by_image_path(data_list, keywords=['PATH', 'RAD']):
    grouped_data = {keyword: {'items': []} for keyword in keywords}
    for item in data_list:
        for keyword in keywords:
            if keyword in item['image']:
                grouped_data[keyword]['items'].append(item)
                break
        else:
            # 如果没有匹配的关键词，默认归类为未知
            if 'unknown' not in grouped_data:
                grouped_data['unknown'] = {'items': []}
            grouped_data['unknown']['items'].append(item)
    
    return grouped_data

# 读取 JSON 文件
with open('modality_laterality_diagnosis_v5_test_all_internvl.json', 'r') as f:
    data_list = json.load(f)
filtered_data = filter_english_questions(data_list)
#grouped_datasets = group_by_image_path(filtered_data,keywords=['PATH', 'RAD','Slake','pretrain','omnimed'])
grouped_datasets = group_by_image_path(filtered_data,keywords=['pretrain'])
all_metrics = {}
for dataset_name, dataset_entries in grouped_datasets.items():
    metrics, closed_metrics, closed_count, closed_correct = calculate_metrics_for_dataset(
        dataset_entries['items']
    )
    print(metrics)
    # Calculate average scores for open-ended questions
    avg_scores = {
        'exact_match_avg': sum(metrics['exact_match']) / len(metrics['exact_match']) if metrics['exact_match'] else 0,
        'f1_avg': sum(metrics['f1']) / len(metrics['f1']) if metrics['f1'] else 0,
        'precision_avg': sum(metrics['precision']) / len(metrics['precision']) if metrics['precision'] else 0,
        'recall_avg': sum(metrics['recall']) / len(metrics['recall']) if metrics['recall'] else 0,
        'closed_accuracy': closed_correct / closed_count if closed_count > 0 else 0
    }
    closed_f1_score_avg = sum(closed_metrics['f1']) / len(closed_metrics['f1'])
    closed_precision_avg = sum(closed_metrics['precision']) / len(closed_metrics['precision'])
    closed_recall_avg = sum(closed_metrics['recall']) / len(closed_metrics['recall'])
    all_metrics[dataset_name] = {
        'open_metrics': avg_scores,
        'closed_f1_score_avg': closed_f1_score_avg,
        'closed_precision_avg': closed_precision_avg,
        'closed_recall_avg': closed_recall_avg,
        'closed_questions': {'count': closed_count, 'correct': closed_correct}
    }

for key in all_metrics:
    print(key,all_metrics[key])
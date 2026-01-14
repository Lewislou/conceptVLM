import json
import collections
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate
import argparse
import warnings

warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for VQA Outputs', add_help=False)
    parser.add_argument('--input', type=str, default="test.json", help='path to input file containing both GT and predictions')
    args, _ = parser.parse_known_args()
    return args

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data 

def calculate_exactmatch(pred, gt):
    return 1 if pred.strip() == gt.strip() else 0

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

def evaluate(data):
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)

    for item in data:
        
        gt_value = item['gt_ans'].lower()
        pred_value = item['answer'].lower()
        question_id = item['question_id']
        question = item['question']
        #print(pred_value, gt_value)
        if 'modality' in question:
            #print(pred_value, gt_value)
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(question_id)

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(question_id)

            b_score = sentence_bleu([gt_value.split()], pred_value.split())
            bleu_scores['q_id'].append(question_id)
            bleu_scores['bleu_score'].append(b_score)
         
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit']) if exact_scores['hit'] else 0.0
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1']) if f1_scores['f1'] else 0.0
    precision = sum(f1_scores['precision']) / len(f1_scores['precision']) if f1_scores['precision'] else 0.0
    recall = sum(f1_scores['recall']) / len(f1_scores['recall']) if f1_scores['recall'] else 0.0
    avg_bleu_score = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score']) if bleu_scores['bleu_score'] else 0.0

    return tabulate(
        [
            ['Exact Match Score', exact_score * 100],
            ['F1 Score', f1_score * 100],
            ['Precision', precision * 100],
            ['Recall', recall * 100],
            ['BLEU Score', avg_bleu_score],
        ],
        headers=['Metric', 'Performance']
    )

if __name__ == '__main__':
    args = parse_option()
    data = load_json(args.input)
    results = evaluate(data)
    print(results)
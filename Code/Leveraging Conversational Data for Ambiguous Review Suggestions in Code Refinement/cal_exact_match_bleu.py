import json
import javalang
from collections import Counter
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from cbleu import *
from smooth_bleu import  bleu_fromstr
def calculate_bleu(reference, candidate):

    reference = [reference.split()]
    candidate = candidate.split()

    bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_score
with open('ground_truth.json', 'r') as f:
  gt_data = json.load(f)
with open('output_fresh.json', 'r') as f:
  output_data = json.load(f)
#with open('src.json', 'r') as f:
#  src_data = json.load(f)
import re

def remove_comments_and_empty_lines(code_str):
    # Remove comments
    code_str = re.sub(r'//.*', '', code_str)  # Remove single-line comments
    code_str = re.sub(r'/\*(.|\n)*?\*/', '', code_str)  # Remove multi-line comments

    # Remove empty lines
    code_str = re.sub(r'\n\s*\n', '\n', code_str)

    return code_str.strip()

def tokenize_code(code_str):
    #code_str = remove_comments_and_empty_lines(code_str)
    tokens = []
    try:
      for token in javalang.tokenizer.tokenize(code_str):
        tokens.append(token.value)
    except:
      return []
    #print(tokens)
    return tokens


def normal_leven2(list1, list2):
    str1 = list1
    str2 = list2
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    matrix = [0 for n in range(len_str1 * len_str2)]

    for i in range(len_str1):
        matrix[i] = i

    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]


def edit_pogress(input, golden, predicted):
    golds = golden
    predictions = predicted
    sources = input
    edit_distance_pred2gold = []
    edit_distance_src2gold = []


    for i in range(len(golds)):
        ## Token Level
        edit_distance_pred2gold.append(
            normal_leven2(golds[i], predictions[i])
        )

        edit_distance_src2gold.append(
            normal_leven2(golds[i], sources[i])
        )

    progress = []
    for i in range(len(edit_distance_pred2gold)):
        pred_ = edit_distance_pred2gold[i]
        src_ = edit_distance_src2gold[1]
        p_ = round((abs(src_) - abs(pred_)) / abs(src_), 3)
        progress.append(p_)
    print(Counter(edit_distance_pred2gold))

    print('Edit Pogress: ', np.sum(np.array(progress)) / len(progress))
    return progress

def min_operations(list1, list2):
    m = len(list1)
    n = len(list2)


    dp = [[(0, '')] * (n + 1) for _ in range(m + 1)]


    for i in range(m + 1):
        dp[i][0] = (i, 'delete')
    for j in range(n + 1):
        dp[0][j] = (j, 'insert')


    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], 'no change')
            else:
                delete = dp[i - 1][j][0] + 1
                insert = dp[i][j - 1][0] + 1
                replace = dp[i - 1][j - 1][0] + 1
                min_steps = min(delete, insert, replace)
                if min_steps == delete:
                    dp[i][j] = (delete, 'delete')
                elif min_steps == insert:
                    dp[i][j] = (insert, 'insert')
                else:
                    dp[i][j] = (replace, 'replace')


    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        operation_type = dp[i][j][1]
        if operation_type == 'no change':
            i -= 1
            j -= 1
        elif operation_type == 'delete':
            operations.append(f"Delete at position {i}")
            i -= 1
        elif operation_type == 'insert':
            operations.append(f"Insert {list2[j-1]} at position {i+1}")
            j -= 1
        elif operation_type == 'replace':
            operations.append(f"Replace {list1[i-1]} with {list2[j-1]} at position {i}")
            i -= 1
            j -= 1

    operations.reverse()
    return operations

exact_match_count = 0
dic_fresh = {}
srcs, gts, preds = [], [] ,[]
bleu_score = 0
gt_strs, pred_strs = [], []
fresh_generates = []
cnt_num = 0
exact_match_ids = []
for i in range(1, len(output_data) - 1):

  if output_data.get(str(i)) == None or output_data.get(str(i)) == "":
      continue
  gt_tokens = tokenize_code(gt_data[str(i)])
  gt_strs.append(' '.join(gt_tokens))
    #print(gt_tokens)
  output_tokens = tokenize_code(output_data[str(i)])
  pred_strs.append(' '.join(output_tokens))
  if gt_tokens == [] or output_tokens == []:
      continue
  bleu_score += nltk_sentence_bleu(' '.join(output_tokens), ' '.join(gt_tokens)) * 100
  cnt_num += 1

  gts.append(gt_tokens)
  preds.append(output_tokens)

  if gt_tokens == output_tokens:
     exact_match_count += 1
     dic_fresh[str(i)] = " ".join(gt_tokens)
     fresh_generates.append(gt_tokens)
     exact_match_ids.append(i)

  else:
     pass


print('Attentd to calculate bleu: ' + str(cnt_num))
with open('fresh_output_chatcot.txt', 'w', encoding='utf-8') as fp:
     for val in fresh_generates:
         fp.write(" ".join(val) + '\n')


pred_nls, golds = pred_strs, gt_strs

for i in range(len(pred_nls)):
    chars = "(_)`."
    for c in chars:
        pred_nls[i] = pred_nls[i].replace(c, " " + c + " ")
        pred_nls[i] = " ".join(pred_nls[i].split())
        golds[i] = golds[i].replace(c, " " + c + " ")
        golds[i] = " ".join(golds[i].split())

print(bleu_score / len(preds))
exact_match = exact_match_count / len(preds)
print('Exact match: {:.2%}'.format(exact_match))
print(exact_match_ids)
#with open('output_fresh.json', 'w') as f_out:
#   json.dump(dic_fresh, f_out, indent=4)
#edit_pogress(srcs, gts, preds)
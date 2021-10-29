# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:46:31 2021

@author: SNT
"""
import os, sys
import numpy as np
sys.path.insert(0, "./metrics/")

from rouge import rouge
from sacreBLEU_script import compute_sacre_bleu
from bleu import compute_bleu

from nltk.translate.meteor_score import meteor_score
from pyter import ter

def compute_metrics(ref_text, hyp_text):
    # INPUTS
    #
    # ref_text: List of reference sentences. Example:
    # ref_text = ['MEHR TRAINING',
    #             'FERTIG HEIM ZUG',
    #             'ABER FÜR HÖREND WIMMELN AUCH SUPER INTEGRATION SCHULE WERFEN']
    # 
    # hyp_text: List of reference sentences. Example:
    # hyp_text =  ['ICH TRAINING NICHT',
    #              'FERTIG ZUG NACH HAUSE',
    #              'AUCH HÖREND GUT INTEGRATION SCHULE']
    #
    # OUTPUT
    #
    # metric_container: A dictionary with the compute score. Example:
    # {'BLEU-GMEAN': 0.515338219427216, 'BLEU-1': 0.7534246575342466,
    # 'BLEU-2': 0.6714285714285714, 'BLEU-3': 0.5970149253731343,
    # 'BLEU-4': 0.53125, 'METEOR': 0.30922723240333894,
    # 'ROUGE_1-F_SCORE': 0.514285709522449, 'ROUGE_1-R_SCORE': 0.537037037037037,
    # 'ROUGE_1-P_SCORE': 0.5444444444444444, 'ROUGE_2-F_SCORE': 0.055555554074074115,
    # 'ROUGE_2-R_SCORE': 0.041666666666666664, 'ROUGE_2-P_SCORE': 0.08333333333333333,
    # 'ROUGE_L-F_SCORE': 0.4310814868187456, 'ROUGE_L-R_SCORE': 0.49999999999999994,
    # 'ROUGE_L-P_SCORE': 0.4777777777777777, 'SACREBLEU': 48.2220701421017,
    # 'TER': 0.8888888888888888}
    #
    
    metric_container = {}
    ref = [[s] for s in ref_text if len(s) > 0]
    hyp = [s for s, rs in zip(hyp_text, ref_text) if len(rs) > 0]
    try:
        'blue-gmean', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
        bleus = compute_bleu(ref, hyp)
        
        metric_container['BLEU-GMEAN'] = bleus[0]
        metric_container['BLEU-1'] = bleus[1][0]
        metric_container['BLEU-2'] = bleus[1][1]
        metric_container['BLEU-3'] = bleus[1][2]
        metric_container['BLEU-4'] = bleus[1][3]    
    except ZeroDivisionError:
        metric_container['BLEU-GMEAN'] = 0
        metric_container['BLEU-1'] = 0
        metric_container['BLEU-2'] = 0
        metric_container['BLEU-3'] = 0
        metric_container['BLEU-4'] = 0          
    
    metric_container['METEOR'] = np.mean([meteor_score(r, h) for r, h in zip(ref, hyp)])

    
    ref = [s for s in ref_text if len(s) > 0]
    rouge_scores = rouge(hyp, ref)
    
    metric_container['ROUGE_1-F_SCORE'] = rouge_scores["rouge_1/f_score"]
    metric_container['ROUGE_1-R_SCORE'] = rouge_scores["rouge_1/r_score"]
    metric_container['ROUGE_1-P_SCORE'] = rouge_scores["rouge_1/p_score"]
    metric_container['ROUGE_2-F_SCORE'] = rouge_scores["rouge_2/f_score"]
    metric_container['ROUGE_2-R_SCORE'] = rouge_scores["rouge_2/r_score"]
    metric_container['ROUGE_2-P_SCORE'] = rouge_scores["rouge_2/p_score"]       
    metric_container['ROUGE_L-F_SCORE'] = rouge_scores["rouge_l/f_score"]        
    metric_container['ROUGE_L-R_SCORE'] = rouge_scores["rouge_l/r_score"]        
    metric_container['ROUGE_L-P_SCORE'] = rouge_scores["rouge_l/p_score"]   

    metric_container["SACREBLEU"] = compute_sacre_bleu(hyp, ref, tokenize = 'char')
    metric_container["TER"] = np.mean([ter(h.split(),r.split()) if len(h) > 0 and len(r) > 0 else 0.0
                                                 for h,r in zip(hyp,ref)])   

    return metric_container 



#%%############################################################################
###                              EXAMPLE OF USAGE                           ###
###############################################################################

# REF_FILENAME = 'sample_text/groundtruth.txt'
# HYP_FILENAME = 'sample_text/transformer_380epochs.txt'
# HYP_FILENAME = 'sample_text/transformer_450epochs.txt'
# HYP_FILENAME = 'sample_text/hyp_mbart.txt'

# with open(REF_FILENAME, 'r', encoding ='utf-8') as f:
#     ref_text = f.read().split('\n')
    
# with open(HYP_FILENAME, 'r', encoding ='utf-8') as f:
#     hyp_text = f.read().split('\n')

# print(compute_metrics(ref_text, hyp_text))

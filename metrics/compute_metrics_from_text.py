# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:31:52 2021

@author: SNT
"""
import os, sys
import numpy as np

import pandas as pd
from bleu import compute_bleu
from rouge import rouge
from WER import wer
from sacreBLEU_script import compute_sacre_bleu
from nltk.translate.meteor_score import meteor_score
from pyter import ter
import numpy as np


WORKING_DIR = 'T2G_TL_NS'
REF_FILENAME = 'groundtruth.txt'

EVAL_EPOCHS = [380]


#%%############################################################################
'''                         ACCOMODATING ES-EN DATA                         '''
###############################################################################

with open(REF_FILENAME, 'r', encoding = 'utf-8') as f:
    ref_data = f.read().split('\n')
    
columns = ['blue-gmean', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
           "rouge_1/f_score", "rouge_1/r_score", "rouge_1/p_score",
           "rouge_2/f_score", "rouge_2/r_score", "rouge_2/p_score",
           "ROUGE-L (F1-score)", "rouge_l/r_score", "rouge_l/p_score",
           'sacreblue', 'METEOR', 'TER']

#%%############################################################################
'''                            COMPUTING METRICS                            '''
###############################################################################    
it_list = []

results_top_container = np.zeros(shape = (len(EVAL_EPOCHS), len(columns)))

imodel = 0
for it in EVAL_EPOCHS:

    with open('{}.txt'.format(it), 'rb') as f:
        generated_text = f.read().decode()
        if "কে"  in generated_text:
            generated_text = generated_text.replace("কে", ' ' ).strip()
        generated_text = [s.strip() for s in generated_text.split('\n')]



    
    ref = [[s] for s in ref_data if len(s) > 0]
    hyp = [s for s, rs in zip(generated_text, ref_data) if len(rs) > 0]
    try:
        bleus = compute_bleu(ref, hyp)
        
        results_top_container[imodel, 0] = bleus[0]
        results_top_container[imodel, 1] = bleus[1][0]
        results_top_container[imodel, 2] = bleus[1][1]
        results_top_container[imodel, 3] = bleus[1][2]
        results_top_container[imodel, 4] = bleus[1][3]    
    except ZeroDivisionError:
        results_top_container[imodel, 0] = 0
        results_top_container[imodel, 1] = 0
        results_top_container[imodel, 2] = 0
        results_top_container[imodel, 3] = 0
        results_top_container[imodel, 4] = 0           
    
    results_top_container[imodel, 15] =  np.mean([meteor_score(r, h) for r, h in zip(ref, hyp)])

    
    ref = [s for s in ref_data if len(s) > 0]
    rouge_scores = rouge(hyp, ref)
    
    results_top_container[imodel, 5] = rouge_scores["rouge_1/f_score"]
    results_top_container[imodel, 6] = rouge_scores["rouge_1/r_score"]
    results_top_container[imodel, 7] = rouge_scores["rouge_1/p_score"]
    results_top_container[imodel, 8] = rouge_scores["rouge_2/f_score"]
    results_top_container[imodel, 9] = rouge_scores["rouge_2/r_score"]
    results_top_container[imodel, 10] = rouge_scores["rouge_2/p_score"]        
    results_top_container[imodel, 11] = rouge_scores["rouge_l/f_score"]        
    results_top_container[imodel, 12] = rouge_scores["rouge_l/r_score"]        
    results_top_container[imodel, 13] = rouge_scores["rouge_l/p_score"]   

    results_top_container[imodel, 14] = compute_sacre_bleu(hyp, ref, tokenize = 'char')
    results_top_container[imodel, 16] = np.mean([ter(h.split(),r.split()) if len(h) > 0 and len(r) > 0 else 0.0
                                                 for h,r in zip(hyp,ref)])   
    
    
    imodel += 1
    
#%%
results = pd.DataFrame(results_top_container, columns = columns, index =EVAL_EPOCHS)
# results.to_excel('{}/metrics.xlsx'.format(WORKING_DIR))


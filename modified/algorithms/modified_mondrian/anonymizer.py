# -*- coding: utf-8 -*-
"""
run modified_mondrian with given parameters
"""
import copy
from pdb import run
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'modified_mondrian'))
from .mondrian import mondrian, mondrian_l_diversity

#sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.data import reorder_columns, restore_column_order


DATA_SELECT = 'a'
DEFAULT_K = 10


def extend_result(val):
    """
    separated with ',' if it is a list
    """
    if isinstance(val, list):
        return ','.join(val)
    return val


<<<<<<< HEAD


def modified_mondrian_anonymize(k, att_trees, data, qi_index, sa_index, ATT_NAMES, Protected_att, goal, outcome, **kwargs):
=======
def modified_mondrian_anonymize(k, att_trees, data, qi_index, sa_index, **kwargs):
>>>>>>> 09f01cb19a511a0e2545f63dff7c7670d6155fcb
    """
    modified Mondrian with K-Anonymity
    """
#    print(data)
    result, runtime = mondrian(
        att_trees, 
        reorder_columns(copy.deepcopy(data), qi_index), 
        k, len(qi_index), sa_index, ATT_NAMES, Protected_att, goal, outcome)

    return restore_column_order(result, qi_index), runtime

def mondrian_ldiv_anonymize(l, att_trees, data, qi_index, sa_index):
    """
    Basic Mondrian with L-diversity
    """
    
    result, runtime = mondrian_l_diversity(
        att_trees, 
        reorder_columns(copy.deepcopy(data), qi_index), 
        l, len(qi_index), sa_index)
    
    
    return restore_column_order(result, qi_index), runtime

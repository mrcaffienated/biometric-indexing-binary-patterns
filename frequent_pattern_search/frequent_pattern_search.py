
from logging.config import valid_ident
from pyexpat import features
import numpy as np
from pathlib import Path
import os
from itertools import product
import operator
import random
from operator import xor
import itertools
import pandas as pd

""" Search of frequent patterns over a single binary representation taking into account the MaxOccurrence.
For patterns with equal MaxOccurrence, it is selected the minimum binary representation.
"""

def generating_K_combinations(length:int):

  """Generate K combinations given a lenght, e.g. 2^4 = 16 binary combinations from 0000 to 1111"""  

  K_combinations = {}

  list_keys = []

  max_code = ""

  for c in enumerate(product(range(2), repeat=length)):

    list_keys.append(str(c[1]))

  K_combinations = dict.fromkeys(list_keys)

  K_combinations = dict.fromkeys(K_combinations.keys(),0)

  return K_combinations


def cleaner_codes(list_max_code):

    """
    cleanner on binary codes 

    """

    list_max_code = [code.strip('()') for code in list_max_code]

    list_max_code = [code.split(',') for code in list_max_code]

    codes_free_space = []
    
    for l in list_max_code:
        
        pattern = ''

        for s in l:

            pattern += s.strip()

        codes_free_space.append(pattern)

    list_max_code = codes_free_space

    return list_max_code


def min_frequent_binary(list_max_code):

    """Return of the minimum binary for equal maximum occurrence, a single frequent patter per subject"""

    # code_dict = {}

    min = 1000000000

    result = 0
    
    for value in list_max_code:

        max_code = int(value,2)

        if max_code < min:

            min = max_code

            result = value

        # code_dict[value] = max_code

    return  result #min(code_dict.keys())



def max_occurrence_search_per_subject_old (bin_dir_feat, length: int):

    """List of codes with maximum occurrence
    
    """

    # binary_feat = np.load(bin_dir_feat)

    binary_feat = bin_dir_feat

    K_combinations = generating_K_combinations(length)

    pos = 0

    max_code = 0

    final_max_code = ''

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i+1
    
    max_value = max(K_combinations.values())

    list_max_code = [k for k,v in K_combinations.items() if v == max_value]

    list_max_code = cleaner_codes(list_max_code)

    if len(list_max_code) > 1:

        final_max_code = min_frequent_binary(list_max_code)

    else:

        final_max_code = list_max_code[0]

    return final_max_code, bin_dir_feat

def max_duplicated_occurrence_search_per_subject(bin_dir_feat, K_combinations, length, examples_on):

    """List of codes with maximum occurrence. Patterns with equal maximum frequency are returned for the binning.
    """
    binary_feat = np.load(bin_dir_feat)

    pos = 0

    max_code = 0

    final_max_code = ''

    array_pos_comb = [0]*len(K_combinations.keys())

    list_max_code = []

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))

        index_pattern = (list(K_combinations.keys())).index(candidate_pattern)

        array_pos_comb[index_pattern] = (array_pos_comb[index_pattern]) + 1

        i = i+1
        
    max_value = max(array_pos_comb)

    list_index_max_code = [index for index, element in enumerate(array_pos_comb) if element == max_value]
    
    list_key = list(K_combinations)

    for index in list_index_max_code:

        list_max_code.append(list_key[index])

    list_max_code = cleaner_codes(list_max_code)

    return list_max_code, bin_dir_feat

def max_occurrence_search_per_subject(bin_dir_feat, K_combinations, length, examples_on):

    """List of codes with maximum occurrence
    
    """

    binary_feat = np.load(bin_dir_feat)

    pos = 0

    max_code = 0

    final_max_code = ''

    array_pos_comb = [0]*len(K_combinations.keys())

    list_max_code = []

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))

        index_pattern = (list(K_combinations.keys())).index(candidate_pattern)

        array_pos_comb[index_pattern] = (array_pos_comb[index_pattern]) + 1

        i = i+1
        
    max_value = max(array_pos_comb)

    list_index_max_code = [index for index, element in enumerate(array_pos_comb) if element == max_value]

    list_key = list(K_combinations)

    for index in list_index_max_code:

        list_max_code.append(list_key[index])

    list_max_code = cleaner_codes(list_max_code)

    if len(list_max_code) > 1:

        final_max_code = min_frequent_binary(list_max_code)

    else:

        final_max_code = list_max_code[0]

    return final_max_code, bin_dir_feat

def sorting_frequent_binaries(list_max_code):

    """Sorting descending of frequent binaries. Return the ranked frequent binaries.
     """

    code_dict = {}

    for value in list_max_code:

        decimal_value = int(value,2)

        code_dict[value] = [decimal_value]

        sorted_code_dict = dict(sorted(code_dict.items(),
                            key=operator.itemgetter(1),
                            reverse=True))

    return sorted_code_dict

def ranked_frequent_patterns(binary_feat_dir, length: int):

    """Return the top of frequent patterns, given a lenght. 
    Suitable for the probe"""

    K_combinations = generating_K_combinations(length)

    binary_feat = np.load(binary_feat_dir)

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i + 1
    
    K_combinations = dict(sorted(K_combinations.items(), key=operator.itemgetter(1),reverse=True))

    list_max_code = [k for k,v in K_combinations.items() if v >= 0]

    list_max_code = cleaner_codes(list_max_code)

    return list_max_code


def random_ranked_frequent_patterns(binary_feat_dir, length: int):

    """Return the top of frequent patterns, given a lenght. 
    Suitable for the probe"""

    K_combinations = generating_K_combinations(length)

    binary_feat = np.load(binary_feat_dir)

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i + 1
    
    K_combinations = dict(sorted(K_combinations.items(), key=operator.itemgetter(1),reverse=True))

    list_max_code = [k for k,v in K_combinations.items() if v >= 0]

    list_max_code = cleaner_codes(list_max_code)

    random.shuffle(list_max_code)

    return list_max_code

def adaptative_ranked_frequent_patterns(binary_feat_dir, length: int):

    """Return the top of frequent patterns, given a lenght. Those patterns with number of frequency > 1 are selected.
    If there exist patterns with the same frequency, these are not sorted suitable for the probe"""

    K_combinations = generating_K_combinations(length)

    binary_feat = np.load(binary_feat_dir)

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i + 1
    
    K_combinations = dict(sorted(K_combinations.items(), key=operator.itemgetter(1),reverse=True))

    list_max_code = [k for k,v in K_combinations.items() if v > 1]

    list_max_code = cleaner_codes(list_max_code)

    return list_max_code


def adaptative_ranked_sorted_similar_frequent_patterns(binary_feat_dir, length: int):

    """Return the top of frequent patterns, given a lenght. Those patterns with number of frequency > 1 are selected.
    If there exist patterns with the same frequency, these are not sorted suitable for the probe"""

    K_combinations = generating_K_combinations(length)

    binary_feat = np.load(binary_feat_dir)

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i + 1
    
    K_combinations = dict(sorted(K_combinations.items(), key=operator.itemgetter(1),reverse=True))

    list_final = {key: occurrence for key, occurrence in K_combinations.items() if occurrence >1}

    list_max_codes = []

    list_max_code_non_similar = []

    list_max_code_similar = []

    list_iter = list(list_final)

    for iter in range(0,len(list_iter)):

        key = list_iter[iter]

        k = [k for k,v in list_final.items() if v == list_final[key]]

        if len(k) == 1:

            k = cleaner_codes(k)

            list_max_codes.append(k[0])
        
        else:

            list_max_code_similar = cleaner_codes(k)

            list_integers = []

            [list_integers.append(int(v,2)) for v in list_max_code_similar]

            list_tuples = list(zip(list_max_code_similar, list_integers))

            list_tuples.sort(key=lambda a: a[1])

            # [list_max_codes.append(c) for c in list_tuple[0]]

            g =2

    return list_final

def random_adaptative_ranked_frequent_patterns(binary_feat_dir, length: int):

    """Return the top of frequent patterns, given a lenght. 
    Suitable for the probe"""

    K_combinations = generating_K_combinations(length)

    binary_feat = np.load(binary_feat_dir)

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

    # for i in range(0, len(binary_feat) - length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i + 1
    
    K_combinations = dict(sorted(K_combinations.items(), key=operator.itemgetter(1),reverse=True))

    list_max_code = [k for k,v in K_combinations.items() if v > 1]

    list_max_code = cleaner_codes(list_max_code)

    random.shuffle(list_max_code)

    return list_max_code

####Ensemble strategy: Concatenation of FBP####
def adaptative_ranked_frequent_patterns_feat_concatenated(binary_feat, length: int):

    """Return the top of frequent patterns, given a lenght. Those patterns with number of frequency > 1 are selected.
    If there exist patterns with the same frequency, these are not sorted suitable for the probe"""

    K_combinations = generating_K_combinations(length)

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))
        
        K_combinations[candidate_pattern] = (K_combinations[candidate_pattern]) + 1

        i = i + 1
    
    K_combinations = dict(sorted(K_combinations.items(), key=operator.itemgetter(1),reverse=True))

    list_max_code = [k for k,v in K_combinations.items() if v > 1]

    list_max_code = cleaner_codes(list_max_code)

    return list_max_code


def max_occurrence_search_per_subject_concatenated_feat(binary_feat, K_combinations, length, examples_on):

    """List of codes with maximum occurrence
    
    """

    pos = 0

    max_code = 0

    final_max_code = ''

    array_pos_comb = [0]*len(K_combinations.keys())

    list_max_code = []

    i = 0

    while (i < (len(binary_feat)-length)) or (len(binary_feat) - i >= length):

        candidate_pattern = binary_feat[i:i+length] 

        candidate_pattern = str(tuple(map(int, candidate_pattern)))

        index_pattern = (list(K_combinations.keys())).index(candidate_pattern)

        array_pos_comb[index_pattern] = (array_pos_comb[index_pattern]) + 1

        i = i+1
        
    max_value = max(array_pos_comb)

    list_index_max_code = [index for index, element in enumerate(array_pos_comb) if element == max_value]

    list_key = list(K_combinations)

    for index in list_index_max_code:

        list_max_code.append(list_key[index])

    list_max_code = cleaner_codes(list_max_code)

    if len(list_max_code) > 1:

        final_max_code = min_frequent_binary(list_max_code)

    else:

        final_max_code = list_max_code[0]

    return final_max_code

########Ensemble strategy based on FBPXor for two biometrics########

def xor_max_multi_modal_2bio(integer_val_b1,integer_val_b2,length):

    bitwise_xor = operator.__xor__(integer_val_b1[0],integer_val_b2[0])
    bin_value = f'{bitwise_xor:0{length}b}'

    return bin_value

########Ensemble strategy based on FBPXor for three biometrics########

def xor_max_multi_modal_3bio(integer_val_b1,integer_val_b2, integer_val_b3, length):

    # result = (A^B) | (B^C)
    bitwise_xor = operator.__xor__(integer_val_b1[0],integer_val_b2[0])
    bitwise_xor_2 = operator.__xor__(integer_val_b2[0],integer_val_b3[0])
    result_or = bitwise_xor | bitwise_xor_2
    bin_value = f'{result_or:0{length}b}'

    return bin_value

########Ensemble strategy based on FBPXor for two biometrics########

def xor_retrieval_multi_modal_2bio(integer_val_b1,integer_val_b2,length):

    list_xor = []
    
    combinations = list(itertools.product(integer_val_b1,integer_val_b2))

    for comb in combinations:
        bitwise_xor = operator.__xor__(comb[0],comb[1])
        bin_value = f'{bitwise_xor:0{length}b}'
        list_xor.append(bin_value)
    
    list_xor_unique = pd.Series(list_xor).drop_duplicates().tolist()

    return list_xor_unique

########Ensemble strategy based on FBPXor for three biometrics########

def xor_retrieval_multi_modal_3bio(integer_val_b1,integer_val_b2,integer_val_b3,length):

    list_xor = []
    
    combinations = list(itertools.product(integer_val_b1,integer_val_b2,integer_val_b3))

    for comb in combinations:
        bitwise_xor = operator.__xor__(comb[0],comb[1])
        bitwise_xor_2 = operator.__xor__(comb[1],comb[2])
        result_or = bitwise_xor | bitwise_xor_2
        bin_value = f'{result_or:0{length}b}'
        list_xor.append(bin_value)
    
    list_xor_unique = pd.Series(list_xor).drop_duplicates().tolist()


    return list_xor_unique


########Ensemble strategy based on FBPMostRanked_Bio for two biometrics########

def ranking_codes_multi_modality_2bio(list_b1, list_b2):

    """Ranked codes across different biometric characteristics
    
    """
    similar_codes = np.intersect1d(list_b1, list_b2)

    ranks_info = []
    codes_info = []

    for code in similar_codes:
        index_b1 = list_b1.index(code)
        index_b2 = list_b2.index(code)
        rank = index_b1 + index_b2
        ranks_info.append(rank)
        codes_info.append(code)

    codes_and_rank = list(zip(codes_info, ranks_info))

    ranked_codes = sorted(codes_and_rank, key = lambda x : x[1],reverse=False)

    sorted_codes,sorted_ranks = zip(*ranked_codes)

    best_code_rank = sorted_codes[0]

    return sorted_codes, best_code_rank

########Ensemble strategy based on FBPMostRanked_Bio for three biometrics########
        
def ranking_codes_multi_modality_3bio(list_b1, list_b2, list_b3):

    """Ranked codes across different biometric characteristics
    
    """
    big_list = [list_b1, list_b2, list_b3]  

    similar_codes = set.intersection(*[set(x) for x in big_list])
    
    # similar_codes = np.intersect1d(list_b1, list_b2, list_b3)

    ranks_info = []

    codes_info = []

    for code in similar_codes:

        index_b1 = list_b1.index(code)

        index_b2 = list_b2.index(code)

        index_b3 = list_b3.index(code)

        rank = index_b1 + index_b2 + index_b3

        ranks_info.append(rank)

        codes_info.append(code)

    codes_and_rank = list(zip(codes_info, ranks_info))

    ranked_codes = sorted(codes_and_rank, key = lambda x : x[1],reverse=False)

    sorted_codes,sorted_ranks = zip(*ranked_codes)

    best_code_rank = sorted_codes[0]

    return sorted_codes, best_code_rank



import argparse
import os
from genericpath import isdir
from collections import defaultdict
from pathlib import Path
from preparing_db import preparing_db
import csv
import numpy as np
from controller.Fusion_Feature_System import Fusion_Feature
from frequent_pattern_search import frequent_pattern_search
import statistics
from pyeer.eer_info import get_eer_stats
from datetime import datetime

parser = argparse.ArgumentParser(description='Generic runner for multi-biometric scheme by weighting FBP')

parser.add_argument('--feat',  '-f',
                    dest="features",
                    type=str,
                    help = 'dir of the features ',
                    default='./data/STOLEN')

parser.add_argument('--bin',  '-b',
                    dest="type_binary",
                    type=str,
                    help =  'type of cancelable scheme applied on embeddings ',
                    default='baseline')

parser.add_argument('--len1',  '-l1',
                    dest="length1",
                    type=int,
                    help =  'length defined to extract FBP over the biometric 1 e.g. face ',
                    default=3)       

parser.add_argument('--numberBins',  '-nbins',
                    dest="numberBins",
                    type=int,
                    help = 'number of bins to filter over probe',
                    default=45)

parser.add_argument('--bio1',  '-b1',
                    dest="bio1",
                    type=str,
                    help = 'Name of the biometric 1',
                    default='Faces')

parser.add_argument('--bio2',  '-b2',
                    dest="bio2",
                    type=str,
                    help = 'Name of the biometric 2',
                    default='Iris')

parser.add_argument('--save',  '-s',
                    dest="save",
                    type=str,
                    help = 'dir save',
                    default='./data/TEST')

args = parser.parse_args()


def compute_scores(k, i, dataset_bio1, dataset_bio2, filter_keys_bio1, filter_keys_bio2, length_1, type_binary, bio1, bio2):
    list_i, list_g = [], []
    print("Round {}".format(i))

    enrolment = Fusion_Feature(length_1, type_binary, 2, bio1, bio2)

    enrol_subjects_bio1, search_subjects_bio1, labels_subjects_bio1, list_dir_feat_ref_bio1  = [],[],[],[]
    enrol_subjects_bio2, search_subjects_bio2, labels_subjects_bio2, list_dir_feat_ref_bio2  = [],[],[],[]

    start = k*i
    end = k*(i + 1)

    impostors_id_bio1 = {k:dataset_bio1[k] for k in filter_keys_bio1[start:end]} #new
    impostors_id_bio2 = {k:dataset_bio2[k] for k in filter_keys_bio2[start:end]} #new

    list_impostors_bio1, list_impostors_bio2 = [],[]

    for imp in impostors_id_bio1:
        list_temp = impostors_id_bio1[imp]
        list_impostors_bio1 = [*list_impostors_bio1, * list_temp]

    for imp in impostors_id_bio2:
        list_temp = impostors_id_bio2[imp]
        list_impostors_bio2 = [*list_impostors_bio2, * list_temp]

    var_iter = set(np.arange(start, end))

    gen_idx = list(set(total_index) - var_iter)
    gen_keys = [filter_keys_bio1[k] for k in gen_idx]

    #getting enrolled subjects of bio1 for each round
    total_id_enrolment = len(gen_keys)
    genuines_id_bio1 = {k:dataset_bio1[k] for k in gen_keys}

    for ref in genuines_id_bio1:
        list_temp = genuines_id_bio1[ref]
        list_dir_feat_ref_bio1 = [*list_dir_feat_ref_bio1, * list_temp]

    enrol_subjects_bio1, search_subjects_bio1, labels_subjects_bio1 = preparing_db.define_protocol(list_dir_feat_ref_bio1,'mix')

    #getting enrolled subjects of bio2 for each round
    gen_keys = [filter_keys_bio2[k] for k in gen_idx]
    genuines_id_bio2 = {k:dataset_bio2[k] for k in gen_keys}

    for ref in genuines_id_bio2:
        list_temp = genuines_id_bio2[ref]
        list_dir_feat_ref_bio2 = [*list_dir_feat_ref_bio2, * list_temp]

    enrol_subjects_bio2,search_subjects_bio2,labels_subjects_bio2 = preparing_db.define_protocol(list_dir_feat_ref_bio2,'mix')

    #loading feats
    enrol_subjects_bio1_load = []
    for e in enrol_subjects_bio1:
        enrol_subjects_bio1_load.append(np.load(e))
    
    enrol_subjects_bio2_load = []
    for e in enrol_subjects_bio2:
        enrol_subjects_bio2_load.append(np.load(e))

    for bio1,label_1,bio2,label_2 in zip(enrol_subjects_bio1_load, labels_subjects_bio1, enrol_subjects_bio2_load, labels_subjects_bio2):
        max_patterns_b1 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(bio1,length_1)
        max_patterns_b2 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(bio2,length_1)
        integer_val_b1 = enrolment.mapping_search(max_patterns_b1)
        integer_val_b2 = enrolment.mapping_search(max_patterns_b2)
        maximum_xor_code = frequent_pattern_search.xor_max_multi_modal_2bio(integer_val_b1,integer_val_b2,length_1) ########Ensemble strategy based on FBPXor########

        enrolment.save_binning_concat_feat_open_set(maximum_xor_code, bio1, bio2, None, 2) #---> closed-set
        
    #loading feats
    search_subjects_bio1_load = []
    for s in search_subjects_bio1:
        search_subjects_bio1_load.append(np.load(s))

    search_subjects_bio2_load = []
    for s in search_subjects_bio2:
        search_subjects_bio2_load.append(np.load(s))


    #starting search process 
    false_negative = 0

    for bio1, label_1, bio2, label_2 in zip(search_subjects_bio1_load, labels_subjects_bio1, search_subjects_bio2_load, labels_subjects_bio2):
        
        list_scores_comp_bio1, list_scores_comp_bio2 = [],[]

        number_comp = 0

        max_patterns_b1 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(bio1,length_1)
        max_patterns_b2 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(bio2,length_1)
        integer_val_b1 = enrolment.mapping_search(max_patterns_b1)
        integer_val_b2 = enrolment.mapping_search(max_patterns_b2)
        xor_codes_retrieval = frequent_pattern_search.xor_retrieval_multi_modal_2bio(integer_val_b1,integer_val_b2,length_1)
        sorted_codes_final = xor_codes_retrieval[0:args.numberBins]
        list_map_codes_s = enrolment.mapping_search(sorted_codes_final)

        for code_search in list_map_codes_s:
            candidate_list = enrolment.search(code_search)

            if len(candidate_list) > 0:
                number_comp+=len(candidate_list)
                list1, list2 = zip(*candidate_list)

                scores_bio1 = enrolment.compare(list1,bio1)
                list_scores_comp_bio1 = [*list_scores_comp_bio1,*scores_bio1]

                scores_bio2 = enrolment.compare(list2,bio2)
                list_scores_comp_bio2 = [*list_scores_comp_bio2,*scores_bio2]

                    
        #here
        list_total_comp.append(number_comp)
        list_scores_comp_bio1.sort()
        list_scores_comp_bio2.sort()

        norm_scores_bio1 = enrolment.normalization_z_score(list_scores_comp_bio1[0], enrolment.normalizer_bio1)
        norm_scores_bio2 = enrolment.normalization_z_score(list_scores_comp_bio2[0], enrolment.normalizer_bio2)

        value_best = norm_scores_bio1 + norm_scores_bio2
        list_g.append(value_best)

    #Compute impostors
    min_impostors = min(len(list_impostors_bio1),len(list_impostors_bio2))
    search_imp_bio1 = list_impostors_bio1[0:min_impostors]
    search_imp_bio2 = list_impostors_bio2[0:min_impostors]

    #loading feat
    search_imp_bio1_load = []
    for s in search_imp_bio1:
        search_imp_bio1_load.append(np.load(s))
    
    search_imp_bio2_load = []
    for s in search_imp_bio2:
        search_imp_bio2_load.append(np.load(s))

    for b1, b2 in zip(search_imp_bio1_load, search_imp_bio2_load):
        list_scores_comp_bio1 ,list_scores_comp_bio2 = [],[]

        max_patterns_b1 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(b1,length_1)
        max_patterns_b2 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(b2,length_1)
        integer_val_b1 = enrolment.mapping_search(max_patterns_b1)
        integer_val_b2 = enrolment.mapping_search(max_patterns_b2)
        xor_codes_retrieval = frequent_pattern_search.xor_retrieval_multi_modal_2bio(integer_val_b1,integer_val_b2,length_1)
        sorted_codes_final = xor_codes_retrieval[0:args.numberBins]
        list_map_codes_s = enrolment.mapping_search(sorted_codes_final)

        for code_search in list_map_codes_s:
            candidate_list = enrolment.search(code_search)

            if len(candidate_list) > 0:
                list1, list2 = zip(*candidate_list)
                
                scores_bio1 = enrolment.compare(list1,b1)
                list_scores_comp_bio1 = [*list_scores_comp_bio1,*scores_bio1]
                
                scores_bio2 = enrolment.compare(list2,b2)
                list_scores_comp_bio2 = [*list_scores_comp_bio2,*scores_bio2]

        #here

        list_scores_comp_bio1.sort()
        list_scores_comp_bio2.sort()

        norm_scores_bio1 = enrolment.normalization_z_score(list_scores_comp_bio1[0], enrolment.normalizer_bio1)
        norm_scores_bio2 = enrolment.normalization_z_score(list_scores_comp_bio2[0], enrolment.normalizer_bio2)

        value_best = norm_scores_bio1 + norm_scores_bio2
        list_i.append(value_best)

    return list_g, list_i

start = datetime.now()
type_binary = args.type_binary
length_1 = args.length1
type_method = args.concat
dir_save = args.save
bio1 = args.bio1
bio2 = args.bio2

#Build path to features
path_feat_bio1 = os.path.join(args.features,'Mix_{}_{}').format(bio1, type_binary)
path_feat_bio2 = os.path.join(args.features,'Mix_{}_{}').format(bio2, type_binary)

features_dir_bin_bio1, features_dir_bin_bio2 = [], []

id_feat_bio1 = os.listdir(path_feat_bio1)

for id in id_feat_bio1:
    if id != '.DS_Store':
        path_dir = os.path.join(path_feat_bio1,id)
        bin_feat = os.listdir(path_dir)

        for bin in bin_feat:
            path_feat = Path(os.path.join(path_dir,bin))
            features_dir_bin_bio1.append(path_feat)

id_feat_bio2 = os.listdir(path_feat_bio2)

for id in id_feat_bio2:

    if id != '.DS_Store':
        path_dir = os.path.join(path_feat_bio2,id)
        bin_feat = os.listdir(path_dir)

        for bin in bin_feat:
            path_feat = Path(os.path.join(path_dir,bin))
            features_dir_bin_bio2.append(path_feat)

dataset_bio1 = preparing_db.preparing_instances_casia(features_dir_bin_bio1)
dataset_bio2 = preparing_db.preparing_instances_casia(features_dir_bin_bio2)

filter_keys_bio1 = [key for key,v in dataset_bio1.items() if len(v) > 1]
filter_keys_bio1.sort()

filter_keys_bio2 = [key for key,v in dataset_bio2.items() if len(v) > 1]
filter_keys_bio2.sort()

k = int(len(filter_keys_bio1)/10)

total_index = list(np.arange(len(filter_keys_bio1)))

indexes_sub = np.arange(len(filter_keys_bio1))

var_index = set(indexes_sub)

permut = np.arange(len(filter_keys_bio1))

path_save_gen = os.path.join(dir_save,'genuines_{}_{}_{}.npy'.format(type_binary,bio1,bio2))

path_save_imp = os.path.join(dir_save,'impostors_{}_{}_{}.npy'.format(type_binary,bio1,bio2))

K_combinations = frequent_pattern_search.generating_K_combinations(length_1)

save_path_csv = os.path.join(dir_save,'{}_{}_{}_order_enrol_{}_{}.csv'.format(type_method, type_binary, length_1, bio1, bio2))

with open(save_path_csv, 'w', newline='') as file:

    fieldnames = ['btp','eer','alpha_best','workload','fmr1000','fmr10000']

    writer_face = csv.DictWriter(file, fieldnames=fieldnames)

    writer_face.writeheader()

    total_id_enrolment = 0
    best_eer = 1000
    alpha = 0
    mated, non_mated = [], []
    list_total_comp = []
    pr_step=0
    best_fmr1000 = 0
    best_fmr10000 = 0

    list_g, list_i = [], []

    for i in range(10):

        gen, imp = compute_scores(k, i, dataset_bio1, dataset_bio2, filter_keys_bio1, filter_keys_bio2, length_1, type_binary, bio1, bio2)
    
        list_g = [*list_g, *gen]
        list_i = [*list_i, *imp]

    stats_a = get_eer_stats(list_g, list_i,ds_scores=True)

    eer = stats_a.eer*100
    best_fmr1000 = stats_a.fmr1000*100
    best_fmr10000 = stats_a.fmr10000*100
    print('Mid EER = {}'.format(eer))
    if eer < best_eer:
        best_eer = eer
        mated = list_g
        non_mated = list_i
        print('Best EER = {}'.format(best_eer))
    
    end = datetime.now()
    time_taken = end - start
    print('Time: ',time_taken) 


    pr = (statistics.mean(list_total_comp)*100)/1008
    pr_step = pr
    print("Workload is {}".format(pr_step))
    
    writer_face.writerow({'btp': type_binary,'eer': best_eer,'workload': pr_step,'fmr1000':best_fmr1000, 'fmr10000':best_fmr10000})

    np.save(path_save_gen, np.asarray(mated))

    np.save(path_save_imp, np.asarray(non_mated))

    print('Best EER = {} and PR per step is {}'.format(best_eer, pr_step))

    

    
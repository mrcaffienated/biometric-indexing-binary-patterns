import argparse
import os
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

parser.add_argument('--feat', '-f', dest="features", type=str, default='./data/STOLEN')
parser.add_argument('--bin', '-b', dest="type_binary", type=str, default='biohashing')
parser.add_argument('--len1', '-l1', dest="length1", type=int, default=3)
parser.add_argument('--numberBins', '-nbins', dest="numberBins", type=int, default=45)
parser.add_argument('--bio1', '-b1', dest="bio1", type=str, default='Faces')
parser.add_argument('--bio2', '-b2', dest="bio2", type=str, default='Iris')
parser.add_argument('--save', '-s', dest="save", type=str, default='./data/TEST')

args = parser.parse_args()


def compute_scores(k, i, dataset_bio1, dataset_bio2,
                   filter_keys_bio1, filter_keys_bio2,
                   length_1, type_binary, bio1, bio2):

    list_i, list_g = [], []
    print(f"Round {i}")

    enrolment = Fusion_Feature(length_1, type_binary, 2, bio1, bio2)

    start = k * i
    end = k * (i + 1)

    impostors_id_bio1 = {k: dataset_bio1[k] for k in filter_keys_bio1[start:end]}
    impostors_id_bio2 = {k: dataset_bio2[k] for k in filter_keys_bio2[start:end]}

    list_impostors_bio1 = []
    list_impostors_bio2 = []

    for imp in impostors_id_bio1:
        list_impostors_bio1.extend(impostors_id_bio1[imp])

    for imp in impostors_id_bio2:
        list_impostors_bio2.extend(impostors_id_bio2[imp])

    var_iter = set(np.arange(start, end))
    gen_idx = list(set(total_index) - var_iter)

    gen_keys_bio1 = [filter_keys_bio1[k] for k in gen_idx]
    gen_keys_bio2 = [filter_keys_bio2[k] for k in gen_idx]

    list_dir_feat_ref_bio1 = []
    list_dir_feat_ref_bio2 = []

    for ref in gen_keys_bio1:
        list_dir_feat_ref_bio1.extend(dataset_bio1[ref])

    for ref in gen_keys_bio2:
        list_dir_feat_ref_bio2.extend(dataset_bio2[ref])

    enrol_subjects_bio1, search_subjects_bio1, labels_subjects_bio1 = \
        preparing_db.define_protocol(list_dir_feat_ref_bio1, 'mix')

    enrol_subjects_bio2, search_subjects_bio2, labels_subjects_bio2 = \
        preparing_db.define_protocol(list_dir_feat_ref_bio2, 'mix')

    enrol_subjects_bio1_load = [np.load(e) for e in enrol_subjects_bio1]
    enrol_subjects_bio2_load = [np.load(e) for e in enrol_subjects_bio2]

    # ENROLMENT
    for bio1_feat, bio2_feat in zip(enrol_subjects_bio1_load, enrol_subjects_bio2_load):

        max_patterns_b1 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(
            bio1_feat, length_1)

        max_patterns_b2 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(
            bio2_feat, length_1)

        _, ranked_code = frequent_pattern_search.ranking_codes_multi_modality_2bio(
            max_patterns_b1, max_patterns_b2)

        enrolment.save_binning_concat_feat_open_set(ranked_code, bio1_feat, bio2_feat, None, 2)

    search_subjects_bio1_load = [np.load(s) for s in search_subjects_bio1]
    search_subjects_bio2_load = [np.load(s) for s in search_subjects_bio2]

    # GENUINE SEARCH
    for bio1_feat, bio2_feat in zip(search_subjects_bio1_load, search_subjects_bio2_load):

        list_scores_comp_bio1 = []
        list_scores_comp_bio2 = []
        number_comp = 0

        max_patterns_b1 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(
            bio1_feat, length_1)

        max_patterns_b2 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(
            bio2_feat, length_1)

        sorted_codes, _ = frequent_pattern_search.ranking_codes_multi_modality_2bio(
            max_patterns_b1, max_patterns_b2)

        sorted_codes_final = sorted_codes[:args.numberBins]
        mapped_codes = enrolment.mapping_search(sorted_codes_final)

        for code in mapped_codes:

            candidate_list = enrolment.search(code)

            if len(candidate_list) > 0:

                number_comp += len(candidate_list)
                list1, list2 = zip(*candidate_list)

                list_scores_comp_bio1.extend(enrolment.compare(list1, bio1_feat))
                list_scores_comp_bio2.extend(enrolment.compare(list2, bio2_feat))

        list_total_comp.append(number_comp)

        list_scores_comp_bio1.sort()
        list_scores_comp_bio2.sort()

        if len(list_scores_comp_bio1) == 0 or len(list_scores_comp_bio2) == 0:
            continue

        norm1 = enrolment.normalization_z_score(
            list_scores_comp_bio1[0], enrolment.normalizer_bio1)

        norm2 = enrolment.normalization_z_score(
            list_scores_comp_bio2[0], enrolment.normalizer_bio2)

        # similarity -> dissimilarity
        value_best = 2 - (norm1 + norm2)
        list_g.append(value_best)

    # IMPOSTOR SEARCH
    min_impostors = min(len(list_impostors_bio1), len(list_impostors_bio2))

    search_imp_bio1_load = [np.load(s) for s in list_impostors_bio1[:min_impostors]]
    search_imp_bio2_load = [np.load(s) for s in list_impostors_bio2[:min_impostors]]

    for b1, b2 in zip(search_imp_bio1_load, search_imp_bio2_load):

        list_scores_comp_bio1 = []
        list_scores_comp_bio2 = []

        max_patterns_b1 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(
            b1, length_1)

        max_patterns_b2 = frequent_pattern_search.adaptative_ranked_frequent_patterns_feat_concatenated(
            b2, length_1)

        sorted_codes, _ = frequent_pattern_search.ranking_codes_multi_modality_2bio(
            max_patterns_b1, max_patterns_b2)

        sorted_codes_final = sorted_codes[:args.numberBins]
        mapped_codes = enrolment.mapping_search(sorted_codes_final)

        for code in mapped_codes:

            candidate_list = enrolment.search(code)

            if len(candidate_list) > 0:

                list1, list2 = zip(*candidate_list)

                list_scores_comp_bio1.extend(enrolment.compare(list1, b1))
                list_scores_comp_bio2.extend(enrolment.compare(list2, b2))

        list_scores_comp_bio1.sort()
        list_scores_comp_bio2.sort()

        if len(list_scores_comp_bio1) == 0 or len(list_scores_comp_bio2) == 0:
            continue

        norm1 = enrolment.normalization_z_score(
            list_scores_comp_bio1[0], enrolment.normalizer_bio1)

        norm2 = enrolment.normalization_z_score(
            list_scores_comp_bio2[0], enrolment.normalizer_bio2)

        value_best = 2 - (norm1 + norm2)
        list_i.append(value_best)

    return list_g, list_i
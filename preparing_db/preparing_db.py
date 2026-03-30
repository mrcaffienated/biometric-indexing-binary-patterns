
from unicodedata import name
import numpy as np
from collections import defaultdict
from random import randrange
import random
import csv
import pandas as pd
import os
from pathlib import Path

""" Define protocol in LFW DB. Return the list of enrolment and search.
 
"""

def preparing_lfw(list_features):

    dataset = {}

    dataset = defaultdict(list)

    for feat_dir in list_features:

        full_name = feat_dir.stem.split('_')

        key=""

        for i in range(0, len(full_name)-1):

            key += full_name[i]+'_' 

        key = key[:-1]

        if key in dataset:

            dataset[key].append(feat_dir)
                    
        else:

            dataset[key] = [feat_dir]

    return dataset


def preparing_SCUT_FVD(list_features):

    dataset = {}

    dataset = defaultdict(list)

    for feat_dir in list_features:

        id = feat_dir.name.split('_')[0]

        if id in dataset:

            dataset[id].append(feat_dir)
                    
        else:

            dataset[id] = [feat_dir]
    
    return dataset

def preparing_instances(list_features):

    db_instances = {}

    db_instances = defaultdict(list)

    for feat_dir in list_features:

        id = feat_dir.name.split('_')[0] + '_' + feat_dir.name.split('_')[1]

        if id in db_instances:

            db_instances[id].append(feat_dir)
                    
        else:

            db_instances[id] = [feat_dir]
    
    return db_instances


def preparing_instances_sdumla(list_features):

    db_instances = {}

    db_instances = defaultdict(list)

    for feat_dir in list_features:

        id = feat_dir.name.split('_')[0] + '_' + feat_dir.name.split('_')[1] + '_' + feat_dir.name.split('_')[2]

        if id in db_instances:

            db_instances[id].append(feat_dir)
                    
        else:

            db_instances[id] = [feat_dir]
    
    return db_instances

def preparing_instances_casia(list_features):

    db_instances = {}

    db_instances = defaultdict(list)

    for feat_dir in list_features:

        id = feat_dir.parent.stem

        # if id.split('_')[1] == 'L': 

        if id in db_instances:

            db_instances[id].append(feat_dir)
                    
        else:

            db_instances[id] = [feat_dir]
        
    return db_instances

#Selecting those subjects containing > 1 sample and reducing 2 samples per subject.
def subset_LFW_selected(list_features,DB):

    enrol_subjects = []

    search_subjects = []

    labels_subjects = []

    if DB == 'LFW':

        dataset = preparing_lfw(list_features)

        for key in dataset:

            if len(dataset[key]) > 1:

                labels_subjects.append(key.strip())

                list_samples = dataset[key]

                # random.shuffle(list_samples)

                enrol_subjects.append(list_samples[0]) 

                search_subjects.append(list_samples[1])

    return enrol_subjects, search_subjects, labels_subjects


def define_protocol(list_features,DB):

    enrol_subjects = []

    search_subjects = []

    labels_subjects = []

    if DB == 'LFW':

        dataset = preparing_lfw(list_features)

        for key in dataset:

            if len(dataset[key]) > 1:

                labels_subjects.append(key.strip())

                list_samples = dataset[key]

                random.shuffle(list_samples)

                enrol_subjects.append(list_samples[0]) 

                search_subjects.append(list_samples[1])
    
    if DB == 'SCUT_FVD':

        dataset = preparing_instances(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            random.shuffle(list_samples)

            enrol_subjects.append(list_samples[0]) 

            search_subjects.append(list_samples[1])
    
    if DB == 'SDUMLA':

        dataset = preparing_instances_sdumla(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            # random.shuffle(list_samples)

            enrol_subjects.append(list_samples[0]) 

            search_subjects.append(list_samples[1])
    
    if DB == 'UTFVP':

         dataset = preparing_instances(list_features)

         for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            random.shuffle(list_samples)

            enrol_subjects.append(list_samples[0]) 

            search_subjects.append(list_samples[1])
    
    if DB == 'CASIA':

        dataset = preparing_instances_casia(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            random.shuffle(list_samples)

            enrol_subjects.append(list_samples[0]) 

            search_subjects.append(list_samples[1])
    
    if DB == 'mix':

        dataset = preparing_instances_casia(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            # random.shuffle(list_samples)

            enrol_subjects.append(list_samples[0]) 

            search_subjects.append(list_samples[1])
            
    return enrol_subjects, search_subjects, labels_subjects

def compute_statistics(list_features,DB):

    enrol_subjects = []

    search_subjects = []

    labels_subjects = []

    list_names_full_enrol = []

    list_names_full_search = []

    if DB == 'mix':

        dataset = preparing_instances_casia(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            # random.shuffle(list_samples)

            enrol_subjects.append(list_samples[0])

            list_names_full_enrol.append(list_samples[0].stem) 

            search_subjects.append(list_samples[1])

            list_names_full_search.append(list_samples[1].stem)
    
    return enrol_subjects,search_subjects,labels_subjects, list_names_full_enrol,list_names_full_search

            



def define_mix_database(list_features):

    list_sdumla = []

    list_utfvp = []

    for f in list_features:

        id = f.parts[-3]

        if 'SDUMLA' in id:

            #process as sdumla

            list_sdumla.append(f)
        
        else:

            list_utfvp.append(f)
    
    dataset = preparing_instances_sdumla(list_sdumla)

    labels_subjects = []

    enrol_subjects = []

    search_subjects = []

    for key in dataset:

        labels_subjects.append(key.strip())

        list_samples = dataset[key]

        random.shuffle(list_samples)

        enrol_subjects.append(list_samples[0]) 

        search_subjects.append(list_samples[1])

    dataset_2 = preparing_instances(list_utfvp)

    labels_subjects_2 = []

    enrol_subjects_2 = []

    search_subjects_2 = []

    for key in dataset_2:

        labels_subjects_2.append(key.strip())

        list_samples = dataset_2[key]

        random.shuffle(list_samples)

        enrol_subjects_2.append(list_samples[0]) 

        search_subjects_2.append(list_samples[1])

    
    list_total_enrol = numbers = [y for x in [enrol_subjects_2,enrol_subjects ] for y in x]

    # list_total_enrol = enrol_subjects + enrol_subjects_2

    list_total_search = search_subjects_2 + search_subjects 

    labels_total_subjects = labels_subjects_2 + labels_subjects 

    return list_total_enrol,list_total_search,labels_total_subjects



def prepare_mix_impostors_instances(list_features):

    list_sdumla = []

    list_utfvp = []

    for f in list_features:

        id = f.parts[-3]

        if 'SDUMLA' in id:

            #process as sdumla

            list_sdumla.append(f)
        
        else:

            list_utfvp.append(f)
    
    dataset = preparing_instances_sdumla(list_sdumla)

    imp_subjects = []

    labels_subjects = []

    for key in dataset:

        labels_subjects.append(key.strip())

        list_samples = dataset[key]

        random.shuffle(list_samples)

        imp_subjects.append(list_samples[0]) 


    dataset_2 = preparing_instances(list_utfvp)

    imp_subjects_2 = []

    labels_subjects_2 = []

    for key in dataset_2:

        labels_subjects_2.append(key.strip())

        list_samples = dataset_2[key]

        random.shuffle(list_samples)

        imp_subjects_2.append(list_samples[0]) 

    list_total_imp = imp_subjects + imp_subjects_2

    list_labels_imp = labels_subjects + labels_subjects_2

    return list_total_imp,list_labels_imp


            

def prepare_impostors_instances(list_features,DB):

    imp_subjects = []

    labels_subjects = []

    if DB == 'SCUT_FVD':

        dataset = preparing_instances(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            random.shuffle(list_samples)

            imp_subjects.append(list_samples[0]) 
    
    if DB == 'SDUMLA':

        dataset = preparing_instances_sdumla(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            random.shuffle(list_samples)

            imp_subjects.append(list_samples[0]) 
    
    if DB == 'UTFVP':

        dataset = preparing_instances(list_features)

        for key in dataset:

            labels_subjects.append(key.strip())

            list_samples = dataset[key]

            random.shuffle(list_samples)

            imp_subjects.append(list_samples[0]) 
    
    return imp_subjects, labels_subjects


def preparing_SDUMLA(list_features):

    dataset = {}

    dataset = defaultdict(list)

    for feat_dir in list_features:

        id = feat_dir.name.split('_')[0] 

        if id in dataset:

            dataset[id].append(feat_dir)
                    
        else:

            dataset[id] = [feat_dir]

    return dataset

def preparing_imp_train_val_sdumla_db(list_features, DB, list_csv):

    df = pd.read_csv (list_csv)

    # print(df)

    names = list(df['idx'])

    list_names = []

    for seq in names:

        filter = seq.split('/')[0] + '_' + seq.split('/')[1] + '_' + seq.split('/')[2]

        filter = filter.split('.bmp')[0] + '.npy'

        list_names.append(filter)

    list_feat_dir_selected = []

    for feat in list_features:

        if feat.name in list_names:

            list_feat_dir_selected.append(feat)

    imp_subjects = []

    labels_subjects = []

    if DB == 'SDUMLA':

        dataset = preparing_instances_sdumla(list_feat_dir_selected)

        for key in dataset:

            list_samples = dataset[key]

            imp_subjects = [*imp_subjects,*list_samples]

            for s in list_samples:

                labels_subjects.append(s.stem)


    return imp_subjects,labels_subjects

def processing_database_finger(path_bin_feat_finger, type_binary):

    features_dir_bin = []

    id_feat = os.listdir(path_bin_feat_finger)

    if type_binary == 'grp' or type_binary == 'urp':

        for id in id_feat:

            if id == '.DS_Store':

                pass
            
            else:

                path_dir = os.path.join(path_bin_feat_finger,id)

                bin_feat = os.listdir(path_bin_feat_finger)

                for bin in bin_feat:

                    path_feat = os.path.join(path_bin_feat_finger,bin)

                    bin_dir = list(Path(path_feat).glob('*npy'))

                    [features_dir_bin.append(e) for e in bin_dir]
    
    else:

        for id in id_feat:

            if id== '.DS_Store':

                pass

            else:

                path_feat = os.path.join(path_bin_feat_finger,id)

                bin_dir = list(Path(path_feat).glob('*npy'))

                [features_dir_bin.append(e) for e in bin_dir]

    return features_dir_bin









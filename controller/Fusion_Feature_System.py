from collections import defaultdict
import numpy as np
from itertools import product
from scipy import spatial


class Fusion_Feature:

    def __init__(self, lenght, type_binary, number_biom_enrol, bio1, bio2, bio3=None):

        self.binning = {}
        self.template_code = {}
        self.map_enrol = defaultdict(list)

        self.lenght = lenght
        self.type_binary = type_binary
        self.number_biom_enrol = number_biom_enrol

        self.bio1 = bio1
        self.bio2 = bio2
        self.bio3 = bio3

        # Default normalizers (safe fallback)
        self.normalizer_bio1 = {'mean': 0, 'std': 1}
        self.normalizer_bio2 = {'mean': 0, 'std': 1}

        self._generate_template_code()

    # -------------------------------------------------
    # TEMPLATE GENERATION
    # -------------------------------------------------

    def _generate_template_code(self):

        for i, c in enumerate(product(range(2), repeat=self.lenght)):
            pattern = ''.join(str(bit) for bit in c)
            self.template_code[pattern] = i

    # -------------------------------------------------
    # ENROL INTERNAL
    # -------------------------------------------------

    def __enrol_multi_binning_concat_feat(self, code, value):

        key_map = self.template_code[code]

        self.binning.setdefault(code, []).append(value)
        self.map_enrol[key_map].append(value)

    # -------------------------------------------------
    # SAVE BINNING (OPEN SET)
    # -------------------------------------------------

    def save_binning_concat_feat_open_set(self, max_code, bio1, bio2, bio3, number_bio):

        if number_bio == 1:
            res = (bio1,)
        elif number_bio == 2:
            res = (bio1, bio2)
        elif number_bio == 3:
            res = (bio1, bio2, bio3)
        else:
            res = ()

        self.__enrol_multi_binning_concat_feat(max_code, res)

    # -------------------------------------------------
    # MAPPING SEARCH
    # -------------------------------------------------

    def mapping_search(self, list_codes):
        return [self.template_code[code] for code in list_codes]

    # -------------------------------------------------
    # SEARCH
    # -------------------------------------------------

    def search(self, mapped_code):

        if mapped_code in self.map_enrol:
            return self.map_enrol[mapped_code]
        return []

    # -------------------------------------------------
    # HAMMING DISTANCE
    # -------------------------------------------------

    def hamming_comparison(self, feat_s, list_feat):
        return [spatial.distance.hamming(feat_e, feat_s) for feat_e in list_feat]

    # -------------------------------------------------
    # GRP SIMILARITY
    # -------------------------------------------------

    def comparison_grp(self, feat_s, list_feat):

        scores = []

        for feat_e in list_feat:
            feat_e = np.array(feat_e)
            feat_s = np.array(feat_s)

            compare = (feat_e == feat_s)
            value = compare.sum() / float(len(feat_e) + len(feat_s) - compare.sum())
            scores.append(value)

        return scores

    # -------------------------------------------------
    # COMPARE (used in your FBPMostRanked_Bio.py)
    # -------------------------------------------------

    def compare(self, list_feat, feat_s):

        if len(list_feat) == 0:
            return []

        if self.type_binary in ['baseline', 'biohashing', 'mlp']:
            scores = self.hamming_comparison(feat_s, list_feat)
            scores.sort()
            return scores

        elif self.type_binary == 'grp':
            scores = self.comparison_grp(feat_s, list_feat)
            scores.sort(reverse=True)
            return scores

        return []

    # -------------------------------------------------
    # NORMALIZATION (SAFE)
    # -------------------------------------------------

    def normalization_z_score(self, item, normalizer):

        if normalizer['std'] == 0:
            return item

        return (item - normalizer['mean']) / normalizer['std']
import os

import warnings

from omnia.generics import np, pd, read_pickle
from omnia.generics.io.reader import Reader
from omnia.proteins import ProteinStandardizer
from omnia.proteins.encoding import NLFEncoder, Esm2Encoder, Esm1bEncoder, ProtbertEncoder, ZScaleEncoder

from .regression_utils import get_keys, extract_data, train_test_split

from Bio import Entrez
from Bio.SeqIO.FastaIO import SimpleFastaParser

from math import floor
from typing import Union
import os
import logging


### REGRESSION ###


warnings.filterwarnings("ignore")


### NOTE ###

# The baseline models used are the same as the ones used in the respective article,
# but the comparison made is NOT the same. In the article, the baseline models used
# serve to evaluate the classification model, which is not created here. So whilst
# the baseline models used are the same, they are used to compare the created regression
# models.

# "The N-terminal subunit, gp120, is completely outside the viral membrane. Although the
# protein has a complex fold it can also be viewed as being linearly organized into five
# conserved regions (C1-C5) interspersed with five variable regions (V1-V5). The host receptor
# CD4 interacts with residues in the conserved regions of gp120 on either side of V4, and the
# coreceptor CCR5 interacts both with a GPGR/Q motif at the apex of the V3 loop and at its base."
# (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3658113/)

# "The CD4 binding site is composed of several discontinuous structural regions of gp120 (ref. 8);"
# (https://www.nature.com/articles/nsmb.1861)

### ---- ###


### Antibody sequences ###

ALL_ABS = {"light":{}, "heavy":{}}

with open(os.path.dirname(__file__) + "/regression_data/light_seqs_aa.fasta", "r") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_ABS["light"][title.split("_")[0]] = seq

with open(os.path.dirname(__file__) + "/regression_data/heavy_seqs_aa.fasta", "r") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_ABS["heavy"][title.split("_")[0]] = seq


### Virus Sequences ###

# --- FETCH VALID VIRUS NAMES AND IDs ---

# ALL_VIR = {}
#
# all_names_ids = []
#
# for line in open("regression_data/viruses.txt", "r").readlines()[1:]:
#     line = line.split("\t")
#     if line[0] != "Virus name": # Check if line is not header line
#         if (line[0] != "") and (line[9] != ""):
#             virus_name = line[0]
#             genbank_id = line[9]
#             all_names_ids.append((virus_name, genbank_id))
#
# vir_names, vir_ids = zip(*all_names_ids)

# vir_ids = {vir_names[ix]: ids for ix, ids in enumerate(vir_ids)}


# --- FETCH VIRUS SEQUENCES ---
#
# Entrez.email = "roberto.bullitta@hotmail.co.uk"
#
# for ix, name in enumerate(vir_names):
#     handle = Entrez.efetch(db="nucleotide", id=vir_ids[name], rettype="gb", retmode="text")
#
#     record = SeqIO.read(handle, "gb")
#     record_feats = [feat for feat in record.features if (feat.type == "CDS") and ("gene" in feat.qualifiers)]
#     env_qual = [feat.qualifiers for feat in record_feats if feat.qualifiers["gene"][0] == "env"]
#     if len(env_qual) > 0:
#         if "translation" in env_qual[0]:
#             ALL_VIR[name] = env_qual[0]["translation"][0]
#         else:
#             print(name, vir_ids[name], env_qual[0])
#     handle.close()
#
#
# --- SAVE SEQS IN FILE ---
#
# with open("regression_data/virus_env_seqs.fasta", "w") as file:
#     to_write = ""
#     for name,seq in ALL_VIR.items():
#         to_write += f">{name} | {vir_ids[name]}\n{seq}\n"
#     file.write(to_write)


# --- RETRIEVE SEQS FROM FILE ---

ALL_VIR = {}
with open(os.path.dirname(__file__) + "/regression_data/virus_env_seqs.fasta") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_VIR[title.split(" | ")[0]] = seq

# ALL_ABS["light"] - 412 (1 extra)
# ALL_ABS["heavy"] - 427 (16 extra)
#
# 411 in common
#
# HEAVY: [3074, 3791, 3869, 3881, 3904, 3BNC55, 4121, CH105, CH235.9, gVRC-H1dC38-VRC01L, VRC13, VRC16, VRC18, VRC38.12,
#         VRC38.13, VRC38.14]
# LIGHT: [2191]


def _get_mean_vals(df:pd.DataFrame) -> pd.DataFrame:
    # df structure:
    # -------------
    # [[1,2,3], [4,5,6]]
    # [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    # [[1,2,3], [4,5,6], [7,8,9]]
    # ...
    # -------------
    # if type(data) == dict:
    #     return {k: [[np.mean(vals) for vals in zip(*v)]] for k, v in data.items()}
    return df.applymap(lambda x: [[np.mean(vals) for vals in zip(*x)]])


def _encoding_dataframe(df, max_len, prefix="heavy"):
    # df structure:
    # -------------
    # [[1,2,3], [4,5,6]]
    # [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    # [[1,2,3], [4,5,6], [7,8,9]]
    # ...
    # -------------
    all_lines = []
    for features in df.iloc[:, 0]:
        line = []
        for ix in range(max_len):
            if ix < len(features):
                line.extend(features[ix])
            else:
                line.extend([0]*len(features[0]))
        all_lines.append(line)

    return pd.DataFrame({f"{prefix}_{i}": v for i, v in enumerate(zip(*all_lines))}, index=df.index)




def extract_complex_encodings(complex_list:np.array, y=None, quantile=0.99, max_sizes=None, encoder="nlf",
                              **kwargs):
    """
    """

    # Store virus env protein sequences and antibody heavy and light chain sequences
    virus = {}
    check_repeated_virus = set()

    abs = {"heavy":{}, "light":{}}
    check_repeated_abs = set()

    complexes = {} # {ab_1: [vir_1, vir_2, ...], ...}

    lines_to_include = []

    for ix, complex in enumerate(complex_list):
        vir, ab = complex.split("__")
        if (vir in ALL_VIR) and (ab in ALL_ABS["heavy"]) and (ab in ALL_ABS["light"]):
            if not (vir in check_repeated_virus):
                virus.update({vir: ALL_VIR[vir]})
                check_repeated_virus.add(vir)
            if not (ab in check_repeated_abs):
                abs["heavy"].update({ab: ALL_ABS["heavy"][ab]})
                abs["light"].update({ab: ALL_ABS["light"][ab]})
                check_repeated_abs.add(ab)

            complexes[ab] = complexes.get(ab, []) + [vir]
            lines_to_include.append(ix)


    if len(complexes) == 0:
        return None

    # Prepare data
    # ProteinStandardizer used to remove ambiguous aminoacids ('X') and replaces other aminoacids
    # (SEE ProteinStandardizer for full info)
    heavy_data = pd.DataFrame({"seq": list(abs["heavy"].values())}, index=list(abs["heavy"].keys()))
    heavy_data, _ = ProteinStandardizer().fit_transform(heavy_data)

    light_data = pd.DataFrame({"seq": list(abs["light"].values())}, index=list(abs["light"].keys()))
    light_data, _ = ProteinStandardizer().fit_transform(light_data)

    virus_data = pd.DataFrame({"seq": list(virus.values())}, index=list(virus.keys()))
    virus_data, _ = ProteinStandardizer().fit_transform(virus_data)

    # Prepare transformer
    if encoder in ["nlf", "protbert", "z-scales"]:
        if encoder == "nlf":
            transformer = NLFEncoder(**kwargs)
        elif encoder == "protbert":
            transformer = ProtbertEncoder(**kwargs)
        else:
            transformer = ZScaleEncoder(**kwargs)

        # Extract heavy and light chain features for each antibody
        heavy_data_features, _ = transformer.fit_transform(heavy_data)

        light_data_features, _ = transformer.fit_transform(light_data)

        virus_data_features, _ = transformer.fit_transform(virus_data)

        # Prepare maximum length values using quantiles and get mean values for 'protbert' encoder
        if encoder == "protbert":
            heavy_data_features = _get_mean_vals(heavy_data_features)
            light_data_features = _get_mean_vals(light_data_features)
            virus_data_features = _get_mean_vals(virus_data_features)
            max_heavy = 1
            max_light = 1
            max_virus = 1

        else:
            max_heavy = floor(np.quantile([len(val) for val in abs["heavy"].values()], quantile))
            max_light = floor(np.quantile([len(val) for val in abs["light"].values()], quantile))
            max_virus = floor(np.quantile([len(val) for val in virus.values()], quantile))
            #print(max_heavy, max_light, max_virus)

    elif ("esm1" in encoder) or ("esm2" in encoder):
        encodings_path = os.path.dirname(os.path.abspath(__file__)) + "/regression_data/encodings/"
        if encoder == "esm1":
            heavy_features_file = "esm1b_t33_650M_UR50S_features_heavy_seqs/features.pkl"
            light_features_file = "esm1b_t33_650M_UR50S_features_light_seqs/features.pkl"
            virus_features_file = "esm1b_t33_650M_UR50S_features_virus_seqs/features.pkl"

        else:
            if not ("pretrained_model" in kwargs):
                kwargs["pretrained_model"] = "8M"

            if kwargs["pretrained_model"] == "8M":
                heavy_features_file = "esm2_t6_8M_UR50D_features_heavy_seqs/features.pkl"
                light_features_file = "esm2_t6_8M_UR50D_features_light_seqs/features.pkl"
                virus_features_file = "esm2_t6_8M_UR50D_features_virus_seqs/features.pkl"
            elif kwargs["pretrained_model"] == "35M":
                heavy_features_file = "esm2_t12_35M_UR50D_features_heavy_seqs/features.pkl"
                light_features_file = "esm2_t12_35M_UR50D_features_light_seqs/features.pkl"
                virus_features_file = "esm2_t12_35M_UR50D_features_virus_seqs/features.pkl"
            elif kwargs["pretrained_model"] == "150M":
                heavy_features_file = "esm2_t30_150M_UR50D_features_heavy_seqs/features.pkl"
                light_features_file = "esm2_t30_150M_UR50D_features_light_seqs/features.pkl"
                virus_features_file = "esm2_t30_150M_UR50D_features_virus_seqs/features.pkl"
            elif kwargs["pretrained_model"] == "650M":
                heavy_features_file = "esm2_t33_650M_UR50D_features_heavy_seqs/features.pkl"
                light_features_file = "esm2_t33_650M_UR50D_features_light_seqs/features.pkl"
                virus_features_file = "esm2_t33_650M_UR50D_features_virus_seqs/features.pkl"
            else:
                heavy_features_file = "esm2_t36_3B_UR50D_features_heavy_seqs/features.pkl"
                light_features_file = "esm2_t36_3B_UR50D_features_light_seqs/features.pkl"
                virus_features_file = "esm2_t36_3B_UR50D_features_virus_seqs/features.pkl"

        #logging.info(msg=encodings_path + heavy_features_file)
        #logging.info(msg=os.getcwd())

        heavy_data = read_pickle(encodings_path + heavy_features_file)
        heavy_data = {k:v for k,v in heavy_data["place_holder"].items() if k in abs["heavy"].keys()} #Get available sequences
        heavy_data_features = pd.DataFrame({"seq": heavy_data.values()}, index=heavy_data.keys())
        heavy_data_features = _get_mean_vals(heavy_data_features)  # Get mean values

        light_data = read_pickle(encodings_path + light_features_file)
        light_data = {k: v for k, v in light_data["place_holder"].items() if k in abs["light"].keys()} #Get available sequences
        light_data_features = pd.DataFrame({"seq": light_data.values()}, index=light_data.keys())
        light_data_features = _get_mean_vals(light_data_features)  # Get mean values

        virus_data = read_pickle(encodings_path + virus_features_file)
        virus_data = {k: v for k, v in virus_data["place_holder"].items() if k in virus.keys()} #Get available sequences
        virus_data_features = pd.DataFrame({"seq": virus_data.values()}, index=virus_data.keys())
        virus_data_features = _get_mean_vals(virus_data_features)  # Get mean values

        # Max length is always 1
        max_heavy = 1
        max_light = 1
        max_virus = 1

    else:
        raise ValueError("'encoder' parameter is not valid. Should be one of the following:\
                          ['nlf', 'esm1', 'esm2-8M', 'esm2-8M', 'esm2-8M', 'esm2-8M', 'esm2-8M', 'protbert', 'z-scales']")

    # Allows continuity for compatibility with preprocessing transformers.
    if max_sizes != None:
        max_heavy, max_light, max_virus = max_sizes

    # Create new columns to allocate each individual feature (with sequence truncating and padding)
    heavy_data_features = _encoding_dataframe(heavy_data_features, max_heavy, prefix="heavy")
    light_data_features = _encoding_dataframe(light_data_features, max_light, prefix="light")
    virus_data_features = _encoding_dataframe(virus_data_features, max_virus, prefix="virus")

    #print(virus_data_features.shape)

    # Join heavy and light chain features
    abs_data_features = pd.concat([heavy_data_features, light_data_features], axis=1)

    #print(abs_data_features.shape)

    # Create final dataset with all complex features
    all_cols = list(abs_data_features.columns) + list(virus_data_features.columns) #26031 features (8677 * 3)
    complex_data = pd.DataFrame(columns=all_cols)


    #print(len(abs))
    #logging.info(msg=str(len(abs)))

    ordered_complex_list = []
    count = 0
    for ab in complexes:
        #logging.info(msg=f"\nAB: {ab} - {len(complexes[ab])} viruses")
        for vir in complexes[ab]:

            line_to_add = list(abs_data_features.loc[ab]) + list(virus_data_features.loc[vir])
            complex_data.loc[count] = line_to_add
            ordered_complex_list.append(vir + "__" + ab)
            count += 1

    # Total complexes: 3662
    #logging.info(msg=f"\n\n{complex_data.shape}\n\n")

    complex_data.index = ordered_complex_list
    if not (y is None):
        return complex_data, y.iloc[lines_to_include, :]
    else:
        return complex_data, y



def convert_complex_to_encodings(input_path="ic50_regression.dat", quantile=0.99, encoder: str = "nfl", seed=42,
                                 **kwargs):

    keys = get_keys(19) #According to figure 2 of respective article
    y = extract_data(input_path, ['IC50'])#.ravel()
    x = extract_data(input_path, keys)

    # Apply log ~ regressor over pIC50
    y = -np.log(y)
    y.index = x.index

    # Split training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True, random_state=seed)

    new_x, new_y = extract_complex_encodings(x.index.values, y, quantile=quantile, encoder=encoder, **kwargs)

    train_inds = [ind for ind in x_train.index if ind in new_x.index.values]
    test_inds =  [ind for ind in x_test.index  if ind in new_x.index.values]

    print(x_train.shape, x_test.shape)

    x_train = new_x.loc[train_inds, :]
    x_test = new_x.loc[test_inds, :]
    y_train = new_y.loc[train_inds, :]
    y_test = new_y.loc[test_inds, :]

    print(x_train.shape, x_test.shape)

    return [x_train, y_train, x_test, y_test]

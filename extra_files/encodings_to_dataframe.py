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
from typing import Union, List, Tuple
import os
import logging


### REGRESSION ###

warnings.filterwarnings("ignore")


### Antibody sequences ###

ALL_ABS = {"light":{}, "heavy":{}}

# Get light-chain sequences
with open(os.path.dirname(__file__) + "/regression_data/light_seqs_aa.fasta", "r") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_ABS["light"][title.split("_")[0]] = seq

# Get heavy-chain sequences
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
    # input df structure:
    # -------------
    # [[1,2,3], [4,5,6]]
    # [[1,2,3], [4,5,6], [7,8,9], [1,2,3]]
    # [[1,2,3], [4,5,6], [7,8,9]]
    # ...
    # -------------

    # output df structure:
    # -------------
    # [[1,2,3]]
    # [[4,5,6]]
    # [[7,8,9]]
    # ...
    # -------------
    return df.applymap(lambda x: [[np.mean(vals) for vals in zip(*x)]])


def _encoding_dataframe(df:pd.DataFrame, max_len:int, prefix:str="heavy") -> pd.DataFrame:
    # input df structure:
    # -------------
    # [[1,2,3], [4,5,6]]
    # [[1,2,3], [4,5,6], [7,8,9], [1,2,3]]
    # [[1,2,3], [4,5,6], [7,8,9]]
    # ...
    # -------------

    # output df structure:
    # -------------
    # [1,2,3,4,5,6,0,0,0,0,0,0]
    # [1,2,3,4,5,6,7,8,9,1,2,3]
    # [1,2,3,4,5,6,7,8,9,0,0,0]
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




def extract_complex_encodings(complex_list:List[str], y:Union[pd.Series, None]=None, quantile:float=0.99,
                              max_sizes:Union[List[int], None]=None, encoder:str="nlf",
                              **kwargs) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Extract the specified encodings for the available light and heavy-chain antibody sequences,
    and viral env protein sequences.

    Parameters
    ----------
    complex_list: List[str]
        Array of complex names (virus_antibody).
    y: Union[pd.Series, None]
        y data.
    quantile: float
        Quantile value to retrieve maximum sequence sizes for each group (light-chain, heavy-chain and viral sequences).
    max_sizes: Union[List[int], None]
        List of predefined maximum sequence sizes (should be of size 3, 1 for each sequence group).
        If None, then the maximum sizes are determined using the defined quantile value.
    encoder: str
        String value to define the type of encodings to extract.
        Options:
            -'nlf' - Uses the NLFEncoder class;
            -'protbert' - Uses the ProtBertEncoder class;
            -'z-scales' - Uses the ZScaleEncoder class;
            -'esm1' - Uses the Esm1bEncoder class;
            -'esm2_8M' - Uses the Esm2Encoder class (pretrained with 8 million parameters);
            -'esm2_35M' - Uses the Esm2Encoder class (pretrained with 35 million parameters);
            -'esm2_150M' - Uses the Esm2Encoder class (pretrained with 150 million parameters);
            -'esm2_650M' - Uses the Esm2Encoder class (pretrained with 650 million parameters);
            -'esm2_3B' - Uses the Esm2Encoder class (pretrained with 3 billion parameters).
    kwargs:
        Arguments to pass to the encoding class used.

    Returns
    -------
    x_data: pd.DataFrame
        The encoding data for all 3 sequence types - light-chain, heavy-chain and viral sequences, respectively.
    y_data: Union[pd.Series, None]
        The 'y' data (if 'y' parameter is not None).
    """

    # Store virus env protein sequences and antibody heavy and light-chain sequences
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

    # Prepare NLF, ProtBert or Z-scale transformer
    if encoder in ["nlf", "protbert", "z-scales"]:
        if encoder == "nlf":
            transformer = NLFEncoder(**kwargs)
        elif encoder == "protbert":
            transformer = ProtbertEncoder(**kwargs)
        else:
            transformer = ZScaleEncoder(**kwargs)

        # Extract heavy and light-chain features for each antibody
        heavy_data_features, _ = transformer.fit_transform(heavy_data)

        light_data_features, _ = transformer.fit_transform(light_data)

        virus_data_features, _ = transformer.fit_transform(virus_data)

        # Prepare maximum length values using quantiles and get mean values for 'protbert' encoder
        if encoder == "protbert":
            heavy_data_features = _get_mean_vals(heavy_data_features)
            light_data_features = _get_mean_vals(light_data_features)
            virus_data_features = _get_mean_vals(virus_data_features)
            # Max length is always 1 for ProtBert encodings
            max_heavy = 1
            max_light = 1
            max_virus = 1

        else:
            max_heavy = floor(np.quantile([len(val) for val in abs["heavy"].values()], quantile))
            max_light = floor(np.quantile([len(val) for val in abs["light"].values()], quantile))
            max_virus = floor(np.quantile([len(val) for val in virus.values()], quantile))
            #print(max_heavy, max_light, max_virus)

    # Prepare ESM-1b or ESM-2 transformer
    elif ("esm1" in encoder) or ("esm2" in encoder):
        # Retrieve pre-extracted encodings from the respective files
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

        # Extract heavy and light-chain encodings for each antibody
        heavy_data = read_pickle(encodings_path + heavy_features_file)
        heavy_data = {k:v for k,v in heavy_data["place_holder"].items() if k in abs["heavy"].keys()} #Get available sequences
        heavy_data_features = pd.DataFrame({"seq": heavy_data.values()}, index=heavy_data.keys())
        heavy_data_features = _get_mean_vals(heavy_data_features)  # Get mean values

        light_data = read_pickle(encodings_path + light_features_file)
        light_data = {k: v for k, v in light_data["place_holder"].items() if k in abs["light"].keys()} #Get available sequences
        light_data_features = pd.DataFrame({"seq": light_data.values()}, index=light_data.keys())
        light_data_features = _get_mean_vals(light_data_features)  # Get mean values

        # Extract env sequence encodings for each virus
        virus_data = read_pickle(encodings_path + virus_features_file)
        virus_data = {k: v for k, v in virus_data["place_holder"].items() if k in virus.keys()} #Get available sequences
        virus_data_features = pd.DataFrame({"seq": virus_data.values()}, index=virus_data.keys())
        virus_data_features = _get_mean_vals(virus_data_features)  # Get mean values

        # Max length is always 1 for ESM-1b and ESM-2 encodings
        max_heavy = 1
        max_light = 1
        max_virus = 1

    else:
        raise ValueError("'encoder' parameter is not valid. Should be one of the following:\
                          ['nlf', 'esm1', 'esm2-8M', 'esm2-35M', 'esm2-150M', 'esm2-650M', 'esm2-3B', 'protbert', 'z-scales']")

    # Allows continuity for compatibility with preprocessing transformers (used in the '4_regression_test_figures.py' file).
    if max_sizes != None:
        max_heavy, max_light, max_virus = max_sizes

    # Create new columns to allocate each individual feature (with sequence truncating and padding)
    heavy_data_features = _encoding_dataframe(heavy_data_features, max_heavy, prefix="heavy")
    light_data_features = _encoding_dataframe(light_data_features, max_light, prefix="light")
    virus_data_features = _encoding_dataframe(virus_data_features, max_virus, prefix="virus")

    # Join heavy and light chain features
    abs_data_features = pd.concat([heavy_data_features, light_data_features], axis=1)

    # Create final dataset with all complex features
    all_cols = list(abs_data_features.columns) + list(virus_data_features.columns) #26031 features (8677 * 3)
    complex_data = pd.DataFrame(columns=all_cols)

    ordered_complex_list = []
    count = 0
    for ab in complexes:
        for vir in complexes[ab]:
            line_to_add = list(abs_data_features.loc[ab]) + list(virus_data_features.loc[vir])
            complex_data.loc[count] = line_to_add
            ordered_complex_list.append(vir + "__" + ab)
            count += 1

    complex_data.index = ordered_complex_list
    if not (y is None):
        return complex_data, y.iloc[lines_to_include, :]
    else:
        return complex_data, y



def convert_complex_to_encodings(input_path:str="ic50_regression.dat", quantile:float=0.99, encoder:str = "nfl",
                                 seed:int=42, **kwargs) -> List[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Extracts specified encodings from given complex data file.

    Parameters
    ----------
    input_path: str
        File path containing antibody-virus complex data with IC50 our breadth values.
    quantile: float
        Quantile value to retrieve maximum sequence sizes for each group (light-chain, heavy-chain and viral sequences).
    encoder: str
        String value to define the type of encodings to extract.
        Possible values are:
            -'nlf' - Uses the NLFEncoder class;
            -'protbert' - Uses the ProtBertEncoder class;
            -'z-scales' - Uses the ZScaleEncoder class;
            -'esm1' - Uses the Esm1bEncoder class;
            -'esm2_8M' - Uses the Esm2Encoder class (pretrained with 8 million parameters);
            -'esm2_35M' - Uses the Esm2Encoder class (pretrained with 35 million parameters);
            -'esm2_150M' - Uses the Esm2Encoder class (pretrained with 150 million parameters);
            -'esm2_650M' - Uses the Esm2Encoder class (pretrained with 650 million parameters);
            -'esm2_3B' - Uses the Esm2Encoder class (pretrained with 3 billion parameters).
    seed: int
        Random seed value.
    kwargs:
        Arguments to pass to the encoding class used.

    Returns
    -------
    x_train: pd.DataFrame
        Complex encoding training data set.
    y_train: pd.Series
        IC50 values of training data set.
    x_test: pd.DataFrame
        Complex encoding test data set.
    y_test: pd.Series
        IC50 values of test data set.
    """

    # Extract complex data from input file
    keys = get_keys(19) #According to figure 2 of respective article
    y = extract_data(input_path, ['IC50'])
    x = extract_data(input_path, keys)

    # Apply log ~ regressor over pIC50
    y = -np.log(y)
    y.index = x.index

    # Split training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True, random_state=seed)

    # Extract encodings for each sequence type (light-chain, heavy-chain and viral sequences)
    new_x, new_y = extract_complex_encodings(x.index.values, y, quantile=quantile, encoder=encoder, **kwargs)

    # Split encodings into train and test sets.
    train_inds = [ind for ind in x_train.index if ind in new_x.index.values]
    test_inds =  [ind for ind in x_test.index  if ind in new_x.index.values]

    x_train = new_x.loc[train_inds, :]
    x_test = new_x.loc[test_inds, :]
    y_train = new_y.loc[train_inds, :]
    y_test = new_y.loc[test_inds, :]

    return [x_train, y_train, x_test, y_test]

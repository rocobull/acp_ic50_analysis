import os

import warnings

from omnia.generics import np,pd
from omnia.proteins import ProteinDescriptor, ProteinStandardizer
from omnia.proteins.encoding import NLFEncoder, Esm2Encoder, Esm1bEncoder, ProtbertEncoder, ZScaleEncoder

from .regression_utils import get_keys, extract_data, train_test_split

from Bio import Entrez
from Bio.SeqIO.FastaIO import SimpleFastaParser

from math import floor



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

# ProteinStandardizer used to remove ambiguous aminoacids ('X') and replaces other aminoacids
# (SEE ProteinStandardizer for full info)

### ---- ###


### Antibody sequences ###

ALL_ABS = {"light":{}, "heavy":{}}

with open(os.path.dirname(__file__) + "/regression_data/light_seqs_aa.fasta", "r") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_ABS["light"][title.split("_")[0]] = seq

with open(os.path.dirname(__file__) + "/regression_data/heavy_seqs_aa.fasta", "r") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_ABS["heavy"][title.split("_")[0]] = seq




ALL_VIR = {}
with open(os.path.dirname(__file__) + "/regression_data/virus_env_seqs.fasta") as handle:
    for title, seq in SimpleFastaParser(handle):
        ALL_VIR[title.split(" | ")[0]] = seq






def encoding_dataframe(df, max_len, prefix="heavy"):
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



def esm_dataframe(df, prefix="heavy"):
    return pd.DataFrame({f"{prefix}_{ix}":v for ix,v in enumerate(zip(*df.iloc[:,0]))}, index=df.index)









def extract_complex_encodings(complex_list:np.array, y=None, quantile=0.99, encoder="nlf",
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

    # heavy_pd = pd.DataFrame({"sequences":abs["heavy"].values()}, index=abs["heavy"].keys())
    # heavy_pd.to_csv("heavy_seqs.csv")
    # light_pd = pd.DataFrame({"sequences": abs["light"].values()}, index=abs["light"].keys())
    # light_pd.to_csv("light_seqs.csv")
    # virus_pd = pd.DataFrame({"sequences": virus.values()}, index=virus.keys())
    # virus_pd.to_csv("virus_seqs.csv")
    # print(3/0)

    if len(complexes) == 0:
        return None

    # Prepare transformer
    if encoder == "nfl":
        transformer = NLFEncoder(**kwargs)
    elif encoder == "esm1":
        transformer = Esm1bEncoder(**kwargs)
    elif "esm2" in encoder:
        transformer = Esm2Encoder(**kwargs)
    elif encoder == "protbert":
        transformer = ProtbertEncoder(**kwargs)
    elif encoder == "z-scales":
        transformer = ZScaleEncoder(**kwargs)
    else:
        raise ValueError("'encoder' parameter is not valid. Should be one of the following:\
                                  ['nlf', 'esm1', 'esm2-8M', 'esm2-8M', 'esm2-8M', 'esm2-8M', 'esm2-8M', 'protbert', 'z-scales']")

    # Extract heavy and light chain features for each antibody
    heavy_data = pd.DataFrame({"seq": list(abs["heavy"].values())}, index=list(abs["heavy"].keys()))
    heavy_data, _ = ProteinStandardizer().fit_transform(heavy_data)
    heavy_data_features, _ = transformer.fit_transform(heavy_data)

    light_data = pd.DataFrame({"seq": list(abs["light"].values())}, index=list(abs["light"].keys()))
    light_data, _ = ProteinStandardizer().fit_transform(light_data)
    light_data_features, _ = transformer.fit_transform(light_data)

    virus_data = pd.DataFrame({"seq": list(virus.values())}, index=list(virus.keys()))
    virus_data, _ = ProteinStandardizer().fit_transform(virus_data)
    virus_data_features, _ = transformer.fit_transform(virus_data)

    # Truncate DataFrame size according to the defined quantile for each sequence type
    max_heavy = floor(np.quantile([len(val) for val in abs["heavy"].values()], quantile))
    max_light = floor(np.quantile([len(val) for val in abs["light"].values()], quantile))
    max_virus = floor(np.quantile([len(val) for val in virus.values()], quantile))

    # Create new columns to allocate each individual feature (with sequence truncating and padding)
    heavy_data_features = encoding_dataframe(heavy_data_features, max_heavy, prefix="heavy")
    light_data_features = encoding_dataframe(light_data_features, max_light, prefix="light")
    virus_data_features = encoding_dataframe(virus_data_features, max_virus, prefix="virus")

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

    #print([val for val in x_train.index.values if "VRC23" in val])
    #print([val for val in new_x.index.values if "VRC23" in val])
    #print(3/0)

    train_inds = [ind for ind in x_train.index if ind in new_x.index.values]
    test_inds =  [ind for ind in x_test.index  if ind in new_x.index.values]

    x_train = new_x.loc[train_inds, :]
    x_test = new_x.loc[test_inds, :]
    y_train = new_y.loc[train_inds, :]
    y_test = new_y.loc[test_inds, :]

    return [x_train, y_train, x_test, y_test]


# if __name__ == "__main__":
#     x_train, y_train, x_test, y_test = convert_complex_to_dataframe()














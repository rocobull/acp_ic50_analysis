import os

import warnings

from omnia.generics import np, pd
from omnia.proteins import ProteinDescriptor, ProteinStandardizer

from .regression_utils import get_keys, extract_data, train_test_split

from Bio import Entrez
from Bio.SeqIO.FastaIO import SimpleFastaParser



### BINARY ###

def convert_acp_to_dataframe(input_path:str) -> pd.DataFrame:
    """
    Extracts data from the "acp740.txt" or "acp240.txt" files and converts them to pd.DataFrame instances.

    Parameters
    ----------
    input_path: str
        The path to the data files.

    Returns
    -------
    data: pd.DataFrame
        The data as a pd.DataFrame instance.
    """
    titles = []
    seqs = []
    y = []

    with open(input_path, "r") as handle:
        for title, seq in SimpleFastaParser(handle):
            titles.append(title)
            seqs.append(seq)
            if "non" in title:
                y.append(0)
            else:
                y.append(1)

    return pd.DataFrame(data={"seq": seqs, "acp": y}, index=titles)

######



### REGRESSION ###

# DISCONTINUED METHOD:
# Using normal descriptors generates terrible results from the models trained.
# It was therefore necessary to use encodings instead (refer to the 'encodings_to_dataframe' file)


# warnings.filterwarnings("ignore")
#
#
# ### Antibody sequences ###
#
# ALL_ABS = {"light":{}, "heavy":{}}
#
# with open(os.path.dirname(__file__) + "/regression_data/light_seqs_aa.fasta", "r") as handle:
#     for title, seq in SimpleFastaParser(handle):
#         ALL_ABS["light"][title.split("_")[0]] = seq
#
# with open(os.path.dirname(__file__) + "/regression_data/heavy_seqs_aa.fasta", "r") as handle:
#     for title, seq in SimpleFastaParser(handle):
#         ALL_ABS["heavy"][title.split("_")[0]] = seq
#
#
# ### Virus Sequences ###
#
# # --- FETCH VALID VIRUS NAMES AND IDs ---
#
# # ALL_VIR = {}
# #
# # all_names_ids = []
# #
# # for line in open("regression_data/viruses.txt", "r").readlines()[1:]:
# #     line = line.split("\t")
# #     if line[0] != "Virus name": # Check if line is not header line
# #         if (line[0] != "") and (line[9] != ""):
# #             virus_name = line[0]
# #             genbank_id = line[9]
# #             all_names_ids.append((virus_name, genbank_id))
# #
# # vir_names, vir_ids = zip(*all_names_ids)
#
# # vir_ids = {vir_names[ix]: ids for ix, ids in enumerate(vir_ids)}
#
#
# # --- FETCH VIRUS SEQUENCES ---
# #
# # Entrez.email = "roberto.bullitta@hotmail.co.uk"
# #
# # for ix, name in enumerate(vir_names):
# #     handle = Entrez.efetch(db="nucleotide", id=vir_ids[name], rettype="gb", retmode="text")
# #
# #     record = SeqIO.read(handle, "gb")
# #     record_feats = [feat for feat in record.features if (feat.type == "CDS") and ("gene" in feat.qualifiers)]
# #     env_qual = [feat.qualifiers for feat in record_feats if feat.qualifiers["gene"][0] == "env"]
# #     if len(env_qual) > 0:
# #         if "translation" in env_qual[0]:
# #             ALL_VIR[name] = env_qual[0]["translation"][0]
# #         else:
# #             print(name, vir_ids[name], env_qual[0])
# #     handle.close()
# #
# #
# # --- SAVE SEQS IN FILE ---
# #
# # with open("regression_data/virus_env_seqs.fasta", "w") as file:
# #     to_write = ""
# #     for name,seq in ALL_VIR.items():
# #         to_write += f">{name} | {vir_ids[name]}\n{seq}\n"
# #     file.write(to_write)
#
#
# ALL_VIR = {}
# with open(os.path.dirname(__file__) + "/regression_data/virus_env_seqs.fasta") as handle:
#     for title, seq in SimpleFastaParser(handle):
#         ALL_VIR[title.split(" | ")[0]] = seq
#
#
#
#
# def extract_complex_descriptors(complex_list:np.array, y=None, preset:str="performance"):
#     """
#     """
#     # Store virus env protein sequences and antibody heavy and light chain sequences
#     virus = {}
#     check_repeated_virus = set()
#
#     abs = {"heavy":{}, "light":{}}
#     check_repeated_abs = set()
#
#     complexes = {} # {ab_1: [vir_1, vir_2, ...], ...}
#
#     lines_to_include = []
#
#     ###
#     exc_virus = set()
#     exc_heavy = set()
#     exc_light = set()
#     ###
#
#     for ix, complex in enumerate(complex_list):
#         vir, ab = complex.split("__")
#         if (vir in ALL_VIR) and (ab in ALL_ABS["heavy"]) and (ab in ALL_ABS["light"]):
#             if not (vir in check_repeated_virus):
#                 virus.update({vir: ALL_VIR[vir]})
#                 check_repeated_virus.add(vir)
#             if not (ab in check_repeated_abs):
#                 abs["heavy"].update({ab: ALL_ABS["heavy"][ab]})
#                 abs["light"].update({ab: ALL_ABS["light"][ab]})
#                 check_repeated_abs.add(ab)
#
#             complexes[ab] = complexes.get(ab, []) + [vir]
#             lines_to_include.append(ix)
#
#     ###
#         if not (vir in ALL_VIR):
#             exc_virus.add(vir)
#         if not (ab in ALL_ABS["heavy"]):
#             exc_heavy.add(ab)
#         if not (ab in ALL_ABS["light"]):
#             exc_light.add(ab)
#
#     print("---")
#     print(len(exc_virus), exc_virus)
#     print(len(exc_heavy), exc_heavy)
#     print(len(exc_light), exc_light)
#     print("---")
#     ###
#
#     if len(complexes) == 0:
#         return None
#
#     # Extract heavy and light chain features for each antibody
#     transformer = ProteinDescriptor(preset=preset)
#
#     # ProteinStandardizer used to remove ambiguous aminoacids ('X') and replaces other aminoacids
#     # (SEE ProteinStandardizer for full info)
#
#     heavy_data = pd.DataFrame({"seq": list(abs["heavy"].values())}, index=list(abs["heavy"].keys()))
#     heavy_data, _ = ProteinStandardizer().fit_transform(heavy_data)
#     heavy_data_features, _ = transformer.fit_transform(heavy_data)
#     heavy_data_features.columns = ["heavy_"+col for col in heavy_data_features]
#
#     light_data = pd.DataFrame({"seq": list(abs["light"].values())}, index=list(abs["light"].keys()))
#     light_data, _ = ProteinStandardizer().fit_transform(light_data)
#     light_data_features, _ = transformer.transform(light_data)
#     light_data_features.columns = ["light_" + col for col in light_data_features]
#
#     abs_data_features = pd.concat([heavy_data_features, light_data_features], axis=1)
#
#     # Extract virus env protein features
#     virus_data = pd.DataFrame({"seq": list(virus.values())}, index=list(virus.keys()))
#     virus_data, _ = ProteinStandardizer().fit_transform(virus_data)
#     virus_data_features, _ = transformer.transform(virus_data)
#     virus_data_features.columns = ["virus_" + col for col in virus_data_features]
#
#
#     # Create final dataset with all complex features
#     all_cols = list(abs_data_features.columns) + list(virus_data_features.columns) #26031 features (8677 * 3)
#     complex_data = pd.DataFrame(columns=all_cols)
#
#     ordered_complex_list = []
#     count = 0
#     for ab in complexes:
#         for vir in complexes[ab]:
#             line_to_add = list(abs_data_features.loc[ab]) + list(virus_data_features.loc[vir])
#             complex_data.loc[count] = line_to_add
#             ordered_complex_list.append(ab + " | " + vir)
#             count += 1
#
#     complex_data.index = ordered_complex_list
#     # print(complex_data.head())
#     # print("Data shape:", complex_data.shape)
#     # print("Total columns:", len(all_cols))
#     # print("Total complexes:", len(complex_list))
#     # print("Removed complexes:", len(complex_list) - complex_data.shape[0])
#     # print()
#     if not (y is None):
#         return complex_data, y.iloc[lines_to_include, :]
#     else:
#         return complex_data, y
#
#
#
# def convert_complex_to_dataframe(input_path="ic50_regression.dat", preset="performance", seed=42):
#
#     keys = get_keys(19) #According to figure 2 of respective article
#     y = extract_data(input_path, ['IC50'])#.ravel()
#     x = extract_data(input_path, keys)
#
#     # Apply log ~ regressor over pIC50
#     y = -np.log(y)
#
#     # Split training and test set
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True, random_state=seed)
#
#     x_train, y_train = extract_complex_descriptors(x_train.index.values, y_train, preset)
#     x_test, y_test = extract_complex_descriptors(x_test.index.values, y_test, preset)
#
#     return [x_train, y_train, x_test, y_test]



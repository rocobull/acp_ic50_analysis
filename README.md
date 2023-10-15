# Developing an automated machine learning framework for protein classification

## Abstract

Researchers are faced with the constant challenge of determining efficient treatments against medically
significant issues, such as cancer and Human Immunodeficiency Virus (HIV), having identified anti-cancer
peptide (ACP)s and antibodies as effective candidates, respectively. The persistent increase in biological
data, however, makes it tough to maintain its curation up to date in order to extract meaningful infor-
mation. This is where the Machine Learning (ML) approaches have proven useful. This field of Artificial
Intelligence (AI) attempts to correlate different sample descriptors with a desired output by applying a
pipeline which includes data preprocessing, algorithm and model selection, and model evaluation. Dif-
ferent tools and frameworks have greatly assisted ML pipeline implementations, such as is the case with
OmniA, an automated ML package in development that allows the implementation of efficient pipelines
with minimal user effort, making it that more practical to apply for even the most inexperienced users.
In this work, OmniA was developed, having been added skewness and multicollinearity treatment trans-
formers, as well as a zero-padding transformer for encodings, along with some bug corrections. OmniA
was then tested against 2 ML/Deep Learning (DL) models used in literature: one using a peptide dataset
to distinguish ACPs from non-ACPs (binary problem), and the other using an antibody datasets for half
maximal inhibitory concentration (IC50) value predictions (regression problem). Code files were prepared
to further automate the ML pipelines, using OmniA’s functionalities, for each prediction problem and gen-
erate results to be used for comparison in regards to the selected articles. The binary prediction results
show that, using only shallow learning models, OmniA was able to produce outstanding results against
the ACP740 dataset, surpassing the comparison article that took a DL approach, and suffering only when
data is more scarce, as is the case with the ACP240 dataset. In regards to the regression problem, by
using protein encoding representations, the results completely surpassed those obtained by the neural
network (NN) approach taken by the authors of the comparison article. Overall, this study proves that Om-
niA is perfectly capable of producing precise predictions, both for binary and regression problems alike,
doing so in an almost completely automated manner and setting the foundation for further automation to
boost the generation of powerful models for biomedical research.

**Keywords:** OmniA, automation, pipeline, preprocessing, proteins.

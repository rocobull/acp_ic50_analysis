from typing import Union

from omnia.generics.scaling import StandardScaler, MinMaxScaler
from omnia.generics.feature_selection import MulticollinearityFS
#from omnia.generics.normalization import OutliersFS
from omnia.generics.normalization import Skewness

from omnia.proteins.feature_extraction.protein_descriptor import ProteinDescriptor



# def _message(type, verbose, *args):
#     """
#     type: #str
#         'x_y_split'
#         'na_remove'
#         'na_treat'
#         'descriptors'
#         'scale'
#         'multicol'
#         'outliers'
#         'skewness'
#     """
#     if verbose:
#
#         if type=="preset":
#             msg = f"{type.upper()}\n{'#'*len(type)}\n"
#
#         elif type=="x_y_split":
#             msg = f"### X-y split ###\n-----------------\n" \
#                   f"Dataset was split into X and y variables\n"
#
#         elif type=="na_remove":
#             msg = f"### NA Removal ###\n------------------\n" \
#                   f"Lines removed: {args[0]}\n"
#         elif type=="na_treat":
#             msg = f"### NA Treatment ({args[1]}) ###\n-----------------------{'-'*len(args[1])}\n" \
#                   f"Lines replaced: {args[0]}\n"
#
#         elif type=="descriptors":
#             msg = f"### Descriptors ###\n-------------------\n" \
#                   f"{args[0]} descriptors added\n"
#
#         elif type=="scale":
#             msg = f"### Scaling ({args[0]}) ###\n------------------\n" \
#                   f"Values scaled\n"
#
#         elif type=="multicol":
#             if len(args[0]) <= 10:
#                 cols = args[0]
#             else:
#                 cols = args[0][:3] + ["..."] + args[0][-3:]
#             msg = f"### Multicolinearity ###\n------------------------\n" \
#                   f"{len(args[0])} Columns removed: {cols}\n"
#
#         elif type=="outliers":
#             msg = f"### Outliers ###\n----------------\n" \
#                   f"Lines removed: {args[0]}\n"
#
#         elif type=="skewness":
#             msg = f"### Skewness ###\n----------------\n" \
#                   f"Columns transformed: {args[0]}\n"
#
#         else:
#             raise ValueError("'type' has an invalid value. Check the class description for permitted values.")
#
#         print(msg)


def preprocess_steps(#X:pd.DataFrame, y:Union[pd.DataFrame,None]=None, problem_type="binary",
               #x_y_split:Union[bool,int,str]=False, y_name:Union[str,int]=-1,
               #na=True, na_treat="mean",
               descriptors:Union[str,None]="performance",
               #outliers=True, outlier_find_method="iqr", outlier_treat_method="capping", outlier_capping_lim=0.9,
               #outlier_flooring_lim=0.1, outlier_multiplier=3, outlier_features_to_consider=None,
               #outlier_features_to_ignore=None,
               skewness=True, skew_upper_limit=1, skew_lower_limit=-1, skew_method="yeo-johnson",
               multicol=True, multicol_corr_limit=0.8, multicol_correlation_method="pearson",
               multicol_features_to_keep=None,
               scale_type:Union[str,None]="standard"):
    """
    """
    steps = []

    # Feature extraction
    if not (descriptors is None):
        steps.append(("descriptor", ProteinDescriptor(preset=descriptors)))

    # # Outliers
    # if outliers:
    #     steps.append(("outliers", OutliersFS(find_method=outlier_find_method, treat_method=outlier_treat_method,
    #                                          capping_limit=outlier_capping_lim, flooring_limit=outlier_flooring_lim,
    #                                          multiplier=outlier_multiplier,
    #                                          features_to_consider=outlier_features_to_consider,
    #                                          features_to_ignore=outlier_features_to_ignore)))

    # Skewness
    if skewness:
        steps.append(("skewness", Skewness(upper_limit=skew_upper_limit,
                                           lower_limit=skew_lower_limit,
                                           method=skew_method)))

    # Multicollinearity
    if multicol:
        if multicol_features_to_keep is None:
            cols_to_keep = []
        else:
            cols_to_keep = multicol_features_to_keep
        steps.append(("multicollinearity", MulticollinearityFS(corr_limit=multicol_corr_limit,
                                                               correlation_method=multicol_correlation_method,
                                                               features_to_keep=cols_to_keep)))

    # Scaling
    if not (scale_type is None):
        if scale_type.lower() == "standard":
            sc = StandardScaler()
        elif scale_type.lower() == "min_max":
            sc = MinMaxScaler()
        else:
            raise ValueError("'scale_type´should be either 'standard' or 'min_max'.")
        steps.append(("scaler", sc))

    return steps

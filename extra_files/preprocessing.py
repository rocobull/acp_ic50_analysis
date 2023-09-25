from typing import Union, List, Tuple, Type

from omnia.generics import Transformer

from omnia.generics.scaling import StandardScaler, MinMaxScaler
from omnia.generics.feature_selection import MulticollinearityFS
from omnia.generics.normalization import Skewness

from omnia.proteins.feature_extraction.protein_descriptor import ProteinDescriptor



def preprocess_steps(descriptors:Union[str, None] = "performance",
               skewness:bool = True, skew_upper_limit:float = 1.0, skew_lower_limit:float = -1.0, skew_method:str = "yeo-johnson",
               multicol:bool = True, multicol_corr_limit:float = 0.8, multicol_correlation_method:str = "pearson",
               multicol_features_to_keep:Union[List[str], None] = None,
               scale_type:Union[str, None] = "standard") -> List[Tuple[str, Type[Transformer]]]:
    """
    Generates a list of preprocessing steps to be used as an input to Omnia's Pipeline class.

    Parameters
    ----------
    descriptors: Union[str, None]
        A descriptor preset (see Omnia's ProteinDescriptor class for more information).
        If None, no ProteinDescriptor instance is added to the list of steps.
    skewness: bool
        Decides whether skewness treatment is added to the list of steps (True) or not (False).
    skew_upper_limit: float
        The upper limit of the Fisher-Pearson coefficient, above or equal
        to which a sample is to be considered skewed.
    skew_lower_limit: float
        The lower limit of the Fisher-Pearson coefficient, below or equal
        to which a sample is to be considered skewed.
    skew_method: str
        The transformation technique used to treat skewed descriptor values.
        Options:
            -'yeo-johnson' - Yeo-Johnson data transformation (for all real values);
            -'box-cox' - Box-Cox data transformation (for positive real values).
    multicol: bool
        Decides whether multicollinearity treatment is added to the list of steps (True) or not (False).
    multicol_corr_limit: float
        Absolute correlation value above which pairs of descriptors are considered collinear.
    multicol_correlation_method: str
        Method for correlation calculation.
        Options:
            -'pearson' - Standard correlation coefficient;
            -'spearman' - Spearman rank correlation;
            -'kendall' - Kendall Tau correlation coefficient.
    multicol_features_to_keep: Union[List[str], None]
        An array of descriptor names to keep, regardless of multicollinearity.
    scale_type: Union[str, None]
        The scaling method to be used.
        Options:
            -'standard' - Standard scaling (uses the StandardScaler class);
            -'min_max' - Min-Max scaling (uses the MinMaxScaler class).
        If None, then the scaling step is not added to the final list of steps.

    Returns
    -------
    steps: List[Tuple[str, Type[Transformer]]]
        A list of tuples, each containing a string value to identify the preprocessing step, and the respective
        preprocessing Transformer.
        If no steps are added, then an empty list is returned.
    """
    steps = []

    # Feature extraction
    if not (descriptors is None):
        steps.append(("descriptor", ProteinDescriptor(preset=descriptors)))

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

import omnia.generics as omnia

from omnia.generics import np, pd

from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef,\
                            mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from typing import Union, List, Tuple
import itertools




def combs(params:dict) -> List[dict]:
    """
    Returns a list of dictionaries with all combinations of the given key-value pairs.

    Parameters
    ----------
    params: dict
        A dictionary containing hyperparameter names as keys, and a list of their possible values as values.

    Returns
    -------
    combinations: List[dict]
        A list of dictionaries, each with a different combination of hyperparameter values.
    """
    val_combinations = itertools.product(*params.values())

    output = []
    for val in val_combinations:
        new_dict = {}
        for k, v in zip(params.keys(), val):
            new_dict[k] = v
        output.append(new_dict)

    return output


def x_y_split(data:pd.DataFrame, y_name:Union[int,str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the independent variables from the dependent variable.

    Parameters
    ----------
    data: pd.DataFrame
        The data.
    y_name: Union[int,str]
        The dependent variable's column name or index.

    Returns
    -------
    X: pd.DataFrame
        The independent variable data.
    y: pd.Series
        The dependent variable data.
    """
    if type(y_name) is int:
        y = pd.DataFrame(data.iloc[:, y_name])
        X = pd.DataFrame(data.iloc[:, data.columns != y.columns])
    elif type(y_name) is str:
        y = pd.DataFrame(data.loc[:, y_name])
        X = pd.DataFrame(data.loc[:, data.columns != y_name])
    else:
        raise TypeError("'y_name' parameter should be either a column index (integer) or name (string).")
    return X, y


def get_binary_scores(y:pd.Series, y_pred:np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Returns the accuracy, recall, specificity, precision and MCC metrics for the 'y' binary value predictions.

    Parameters
    ----------
    y: pd.Series
        The true 'y' values.
    y_pred
        The predicted 'y' values.

    Returns
    -------
    scores: Tuple[float, float, float, float, float]
        The accuracy, recall, specificity, precision and MCC metric values
    """
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    specificity = recall_score(y, y_pred, pos_label=0) #Same as recall, but for negative samples
    precision = precision_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    # predictions = [y.values[i] == y_pred[i] for i in range(len(y))]

    # positive_mask = [i for i, val in enumerate(y.values) if (int(val) == 1)]
    # negative_mask = [i for i, val in enumerate(y.values) if (int(val) == 0)]

    # TP = sum([1 if val else 0 for i, val in enumerate(predictions) if (i in positive_mask)]) #VP
    # TN = sum([1 if val else 0 for i, val in enumerate(predictions) if (i in negative_mask)]) #VN
    # FP = sum([1 if not val else 0 for i, val in enumerate(predictions) if (i in negative_mask)])
    # FN = sum([1 if not val else 0 for i, val in enumerate(predictions) if (i in positive_mask)])

    # accuracy = (TN+TP)/(TN+TP+FP+FN)
    # recall = TP/(FN+TP)
    # specificity = TN/(TN+FP)
    # precision = TP/(TP+FP)
    # mcc = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return accuracy, recall, specificity, precision, mcc



def get_regression_scores(y:pd.Series, y_pred:np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Returns the RMSE, MSE, r2, Pearson correlation and Spearman correlation metrics for the 'y' real value predictions.

    Parameters
    ----------
    y: pd.Series
        The true 'y' values.
    y_pred
        The predicted 'y' values.

    Returns
    -------
    scores: Tuple[float, float, float, float, float]
        The RMSE, MSE, r2, Pearson correlation and Spearman correlation metric values
    """
    rmse = mean_squared_error(y, y_pred, squared=False)
    mse = mean_squared_error(y, y_pred, squared=True)
    r2 = r2_score(y, y_pred)
    pearson = pearsonr(y, y_pred)[0]
    spearman = spearmanr(y, y_pred)[0]

    return rmse, mse, r2, pearson, spearman



MODELS =[
        omnia.RandomForestModel,
        omnia.MultilayerPerceptronNN,
        omnia.CatBoostModel,
                #omnia.FastTextModel,
        omnia.KNNModel,
        omnia.LGBModel,
        omnia.LinearModel,
        omnia.FastAINN,
        omnia.VowpalWabbitModel,
        omnia.XGBoostModel,
        omnia.XTModel,
        omnia.SupportVectorMachineModel
]

VARIABLE_PARAMS = combs({
                    "skewness":[True,False],
                    "scale_type":["standard", "min_max"],
                    "multicol":[True,False]
                    })


GENERAL_PARAMS = {"skew_upper_limit":1, "skew_lower_limit":-1, "skew_method":"yeo-johnson",
                  "multicol_correlation_method": "spearman", "multicol_corr_limit":0.8, "multicol_features_to_keep":["length"]
                 }


PRESETS = ['all', 'performance', 'physico-chemical']
           #'aac',
           #'paac', 'auto-correlation', 'composition-transition-distribution',
                #'seq-order',
           #'modlamp-correlation', 'modlamp-all']

# for params in VARIABLE_PARAMS:
#    print(params)
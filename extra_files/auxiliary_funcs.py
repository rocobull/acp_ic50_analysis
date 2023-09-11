import omnia.generics as omnia

from omnia.generics import np, pd

from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef,\
                            mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from typing import Union
import itertools





# def _check_repeats(param_dict):
#     """
#
#     """
#     if "multicol" in param_dict:
#         if "multicol_correlation_method" in param_dict:
#             if (param_dict["multicol"] is False) and (param_dict["multicol_correlation_method"].lower() != "pearson"):
#                 return False
#
#     if "outliers" in param_dict:
#         if "outlier_treat_method" in param_dict:
#             if (param_dict["outliers"] is False) and (param_dict["outlier_treat_method"].lower() != "capping"):
#                 return False
#     #...
#     return True

def combs(params):
    """
    Returns a list of dictionaries with all combinations of the given key-value pairs.
    """
    val_combinations = itertools.product(*params.values())

    output = []
    for val in val_combinations:
        new_dict = {}
        for k,v in zip(params.keys(), val):
            new_dict[k] = v
        #if _check_repeats(new_dict):
        output.append(new_dict)

    return output


def x_y_split(data:pd.DataFrame, y_name:Union[int,str]):
    """

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


def get_binary_scores(y:pd.Series, y_pred:np.ndarray):

    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    specificity = recall_score(y, y_pred, pos_label=0) #Same as recall, but for negative samples
    precision = precision_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    # predictions = [y.values[i] == y_pred[i] for i in range(len(y))]
    #
    # positive_mask = [i for i, val in enumerate(y.values) if (int(val) == 1)]
    # negative_mask = [i for i, val in enumerate(y.values) if (int(val) == 0)]
    #
    # TP = sum([1 if val else 0 for i, val in enumerate(predictions) if (i in positive_mask)]) #VP
    # TN = sum([1 if val else 0 for i, val in enumerate(predictions) if (i in negative_mask)]) #VN
    # FP = sum([1 if not val else 0 for i, val in enumerate(predictions) if (i in negative_mask)])
    # FN = sum([1 if not val else 0 for i, val in enumerate(predictions) if (i in positive_mask)])
    #
    # accuracy = (TN+TP)/(TN+TP+FP+FN)
    # recall = TP/(FN+TP)
    # specificity = TN/(TN+FP)
    # precision = TP/(TP+FP)
    # mcc = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return accuracy, recall, specificity, precision, mcc



def get_regression_scores(y:pd.Series, y_pred:np.ndarray):

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

for params in VARIABLE_PARAMS:
   print(params)
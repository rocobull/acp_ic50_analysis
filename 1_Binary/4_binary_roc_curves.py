from sklearn.metrics import roc_curve, auc
from Bio.SeqIO.FastaIO import SimpleFastaParser

from omnia.generics import pd
from extra_files import convert_acp_to_dataframe, x_y_split

import matplotlib.pyplot as plt

from typing import List



def plot_roc_curve(labels:List[float], probability:List[float], filename:str,
                   fig_title:str = "ROC", legend_text:str = 'proposed method') -> None:
    """
    Creates and saves a ROC curve, along with the calculated AUC value,
    from the true 'y' and predicted 'y' probability values.

    Parameters
    ----------
    labels: List[float]
        List of true 'y' values.
    probability: True[float]
        List of predicted 'y' probability values.
    filename: str
        Name of the file to save the ROC curve.
    fig_title: str
        Title of the ROC curve figure.
    legend_text: str
        Legend of the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(labels, probability)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color="orange", label=legend_text + ' (AUC=%6.3f) ' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fig_title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.clf()
    #plt.show()


# Iterate over the file names
for name in ["740", "240"]:

    # Get true 'y' values in the same order as the test folds
    if name == "740":
        data = convert_acp_to_dataframe("acp740.txt")
    else:
        data = convert_acp_to_dataframe("acp240.txt")

    true_vals = []

    for fold in range(5):
        test = pd.DataFrame([data.iloc[i, :] for i, _ in enumerate(data.index) if i % 5 == fold], columns=data.columns)
        _, y_test = x_y_split(test, "acp")
        true_vals.extend(list(y_test.iloc[:, 0]))

    # Retrieve the probability predictions from the created files and generate the respective ROC curve graphs.
    with open(f'{name}_test_predictions.txt', "r") as handle:
        for title, pred_vals in SimpleFastaParser(handle):
            pred_vals = [float(val) for val in pred_vals.strip().split("\t")]
            title_split = title.split("_")
            plot_roc_curve(true_vals, pred_vals,
                           filename=f"{name}_{title}",
                           fig_title=f"{name}: {' / '.join(title_split)}",
                           legend_text=title_split[0])

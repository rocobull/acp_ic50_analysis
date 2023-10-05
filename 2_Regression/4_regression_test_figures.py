import sys
sys.path.append("/home/rbullita/Thesis/omnia/omnia-proteins/examples/Pipelines")
# Important to define 'Pipelines' as a source file to function in virtual server

from extra_files import convert_complex_to_encodings, extract_complex_encodings,\
                        preprocess_steps,\
                        GENERAL_PARAMS, VARIABLE_PARAMS,\
                        seaman2010, antibodies, breadth_exp

from omnia.generics import CatBoostModel, Model, Transformer, np

from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import scipy
import random

import logging

#logging.basicConfig(filename="graphs_logs.log", filemode="a", level="INFO")



def make_fig2(model_name:str, model_hyperparams:dict, preset:str,
              params:dict, comb:dict, random_state:int=np.random) -> None:
    """
    Recreates Figure 2 from Conti S. & Karplus M. (2019) (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954).

    Parameters
    ----------
    model_name: str
        The name of the model to be used.
    model_hyperparams: dict
        The hyperparameters of the model to be used.
    preset: str
        The encodings preset to extract for each sequence type.
    params: dict
        Arguments to pass to the encoding class used.
    comb: dict
        The variable hyperparameters to use for the preprocessing steps extraction.
    random_state: int
        Random seed value.
    """
    # Split data and extract encodings
    X_train, y_train, X_test, y_test = convert_complex_to_encodings('ic50_regression.dat', quantile=0.99,
                                                                    encoder=preset, seed=random_state, **params)
    # Get the preprocesisng steps
    steps = preprocess_steps(descriptors=None,
                             **GENERAL_PARAMS,
                             **comb)

    # Fit the transformer of each step to the training data and transform training and testing data
    for _, transformer in steps:
        transformer.fit(X_train, y_train)
        X_train, y_train = transformer.transform(X_train, y_train)
        X_test, y_test = transformer.transform(X_test, y_test)

    # for column in X_train:
    #     X_train[column] = np.nan_to_num(X_train[column].astype(np.float32))
    #     X_test[column] = np.nan_to_num(X_test[column].astype(np.float32))

    # In this work, only the "CatBoostModel" was determined as the best performing model
    if "CatBoostModel" in model_name:
        # The following values must be converted to match those of the input parameters of the model class
        # (Discrepancies in parameter names between Autogluon and Omnia)
        model_hyperparams["eval_metric"] = "rmse"
        predictor = CatBoostModel(**model_hyperparams)

    # # 'eval_metric' is not defined in the _init_parameters dictionary otherwise:
    # predictor._init_parameters = model_hyperparams

    # Fit the model and get training and testing data predictions
    predictor.fit(X_train, y_train)
    y_train_predict = predictor.predict(X_train)
    y_test_predict = predictor.predict(X_test)

    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    # Make figure
    plt.plot(y_train, y_train_predict, '.b', label='Train set')
    plt.plot(y_test, y_test_predict, '.r', color="orange", label='Test set')
    plt.ylim(-5, 5)
    plt.ylim(-5, 10)
    plt.title(f'pIC50 - {preset}')
    plt.xlabel('Experimental')
    plt.ylabel('Computed')
    plt.legend()
    plt.savefig(f'fig2_{preset}.png', bbox_inches='tight', dpi=300)
    plt.clf()

    with open(f'fig2_{preset}_data.dat', 'w') as fp:

        # Check accuracy in train set
        rp = scipy.stats.pearsonr(y_train, y_train_predict)[0]
        rs = scipy.stats.spearmanr(y_train, y_train_predict)[0]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_train, y_train_predict)
        fp.write('Correlation in training set\n')
        fp.write("Pearson   : %.3f\n" % (rp))
        fp.write("Spearman  : %.3f\n" % (rs))
        fp.write("Slope     : %f\n" % (slope))
        fp.write("Intercept : %f\n\n" % (intercept))

        # Check accuracy in test set
        rp = scipy.stats.pearsonr(y_test, y_test_predict)[0]
        rs = scipy.stats.spearmanr(y_test, y_test_predict)[0]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, y_test_predict)
        fp.write('Correlation in validation set\n')
        fp.write("Pearson   : %.3f\n" % (rp))
        fp.write("Spearman  : %.3f\n" % (rs))
        fp.write("Slope     : %f\n" % (slope))
        fp.write("Intercept : %f\n\n" % (intercept))




def eval_model(abname:str, agname:str, preset:str, params:dict, model:Model,
               steps:List[Tuple[str, Transformer]], max_sizes:List[int, int, int]) -> Union[float, None]:
    """
    Get pIC50 predictions from given antibody-antigen complex (if existent).

    Parameters
    ----------
    abname: str
        The name of the antibody to study.
    agname: str
        The name of the antigen to study.
    preset: str
        The encodings preset to extract for each sequence type in the complex.
    params: dict
        Arguments to pass to the encoding class used.
    model: Model
        The model instance to use for pIC50 predictions
    steps: List[Tuple[str, Transformer]]
        A list of tuples, each containing a string value to identify the preprocessing step, and the respective
        preprocessing Transformer.
    max_sizes: List[int, int, int]
        List of predefined maximum sequence sizes (should be of size 3, 1 for each sequence group).

    Returns
    -------
    calc: Union[float, None]
        The calculated pIC50 value if the complex is existent.
        Returns None if otherwise.
    """
    # Predefined by the authors of the comparison article
    # (Conti S. & Karplus M. (2019)) (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954).
    if agname in ['271_11', 'T266_60', 'T33_7']:
        return None

    # Extract encodings for the given antibody-antigen complex
    cpxname = '%s__%s' % (agname, abname)
    data = extract_complex_encodings([cpxname], max_sizes=max_sizes, encoder=preset, **params)

    # preprocess data (if existent) and get pIC50 calculations
    if data:  # Check if data_line is not None
        data_line = data[0]
        for step, transformer in steps:
            data_line, _ = transformer.transform(data_line)
        data_line.columns = [str(col) for col in data_line.columns]
        calc = model.predict(data_line)
        return calc
    else:
        return None


def compute_breadth(abname:str, model:Model, preset:str, params:dict,
                    steps:List[Tuple[str, Transformer]], max_sizes:List[int, int, int]) -> Tuple[int, int, float]:
    """
    Compute the breadth of an antibody given the model to use.

    Parameters
    ----------
    abname: str
        The name of the antibody to study.
    model: Model
        The model instance to use for pIC50 predictions
    preset: str
        The encodings preset to extract for each sequence type in the complex.
    params: dict
        Arguments to pass to the encoding class used.
    steps: List[Tuple[str, Transformer]]
        A list of tuples, each containing a string value to identify the preprocessing step, and the respective
        preprocessing Transformer.
    max_sizes: List[int, int, int]
        List of predefined maximum sequence sizes (should be of size 3, 1 for each sequence group).

    Returns
    -------
    breadth_vals: Tuple[int, int, float]
        A tuple containing the total number of complexes, the number of IC50 values below a determined threshold,
        and the breadth value.
    """
    br = 0
    tot = 0
    # Determine antibody breadth against the Seaman virus panel
    for vname in seaman2010:
        ic50 = eval_model(abname, vname, preset, params, model, steps, max_sizes)
        if ic50 is not None:
            br += ic50 >= 1 #Threshold defined by the authors
            tot += 1
    return tot, br, br/tot


def compute_breadth_all_average(model_name:str, model_hyperparams:dict, preset:str, params:dict, comb_index:int,
                                ablist:List[str], fname:str="ic50_regression.dat",
                                ntrials:int=30, random_state:int=np.random) -> dict:
    """
    Compute the breadth for all antibodies in ablist over a number of iterations (defined by the 'ntrials' parameter).

    Parameters
    ----------
    model_name: str
        The name of the model to be used.
    model_hyperparams: dict
        The hyperparameters of the model to be used.
    preset: str
        The encodings preset to extract for each sequence type in the complex.
    params: dict
        Arguments to pass to the encoding class used.
    comb_index: int
        The index of the predefined VARIABLE_PARAMS dictionary,
        containing different 'preprocess_steps' function hyperparameter combinations to test.
    ablist: str
        A list of antibody names.
    fname: str
        The file name from which to extract data for initial model training.
    ntrials: int
        Number of different data split iterations to train the model with for pIC50 value predictions for each antibody.
    random_state: int
        Random seed value.
    Returns
    -------
    A tuple containing the total number of complexes, the number of IC50 values below a determined threshold,
    and the breadth value.
    """

    # Prepare dictionary to store antibody names and their predicted breadth values over 'ntrials' iterations
    allbreadth = {ab: list() for ab in ablist}

    # Perform 'ntrial' iterations
    for i in range(ntrials):
        # Obtain unique training data splits and generate their encodings
        X_train, y_train, X_test, y_test = convert_complex_to_encodings(input_path=fname, quantile=0.99, encoder=preset,
                                                                        seed=random_state*i, **params)

        # Determine fixed size of each sequence type to maintain coherence when extracting encodings for individual complexes
        if preset == "z-scales":
            max_sizes = [sum(1 for col in X_train.columns if col.startswith(suffix)) for suffix in ["heavy_", "light_", "virus_"]]
            max_sizes = [int(val / 5) for val in max_sizes] # (number of z-scale values for each amino acid, to get the fixed size of each sequence type)
            #print(max_sizes)
        else:
            max_sizes = [1, 1, 1] # (For esm2_3B, the mean values are calculated for each amino acid, so there is only 1 encoding list per sequence type)

        # Get preprocessing combination and the respective preprocessing steps
        comb = VARIABLE_PARAMS[int(comb_index)]
        steps = preprocess_steps(descriptors=None,
                                 **GENERAL_PARAMS,
                                 **comb)

        # Manually transform training data
        for step, transformer in steps:
            X_train, y_train = transformer.fit_transform(X_train, y_train)

        X_train.columns = [str(col) for col in X_train.columns]

        # In this work, only the "CatBoostModel" was determined as the best performing model
        if "CatBoostModel" in model_name:
            # The following values must be converted to match those of the input parameters of the model class
            # (Discrepancies in parameter names between Autogluon and Omnia)
            model_hyperparams["eval_metric"] = "rmse"
            predictor = CatBoostModel(**model_hyperparams)

        # Fit the model
        predictor.fit(X_train, y_train)

        # Iterate over each antibody in the given antibody list and obtain their respective breadth predictions
        for abname in ablist:
            logging.info(msg=f"\t### {abname} ###")
            _, _, breadth = compute_breadth(abname, predictor, preset, params, steps, max_sizes)
            allbreadth[abname].append(breadth)

    # Get mean and standard deviations for all predicted breadth values of each antibody
    result = dict()
    for abname in ablist:
        avg = np.mean(allbreadth[abname])
        std = np.std(allbreadth[abname])
        result[abname] = (avg, std)
    return result




def make_fig4(breadth_calc:dict, title:str, fname:str):
    """
    Recreates Figure 4 from Conti S. & Karplus M. (2019) (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954).

    Parameters
    ----------
    breadth_calc: dict
        A dictionary containing antibody names as keys and a tuple containing the mean and standard deviations
        of their predicted breadth values as values.
    title: str
        The figure's title name.
    fname: str
        The file name to save the figure.
    """
    exp = breadth_exp # Experimental breadth of the antibodies list
    avg = list()
    std = list()
    for ab in antibodies:
        avg.append(breadth_calc[ab][0])
        std.append(breadth_calc[ab][1])
    slope, intercept, _, _, _ = scipy.stats.linregress(exp, avg)
    plt.plot(np.linspace(-1,2,20), intercept+slope*np.linspace(-1,2,20), 'k-')
    rp = scipy.stats.pearsonr(exp, avg)[0]
    rs = scipy.stats.spearmanr(exp, avg)[0]
    plt.plot(exp, avg, 'b.', color="orange", label="rp=%.3f\nrs=%.3f" % (rp, rs))
    plt.errorbar(exp, avg, yerr=std, fmt='b.', ecolor="orange")
    plt.axis('equal')
    plt.axis((-0.05,1.05,-0.05,1.05))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Experimental breadth')
    plt.ylabel('Calculated breadth')
    plt.legend()
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.clf()





# Get the chosen models' hyperparameters
all_hyperparams = []

for line in open("model_hyperparameters.txt", "r").readlines():

    # If line is an entry title:
    if line[0] == ">":
        model_name = line.strip().split(" ")[1].split("_")[-1]
        hyperparams = [model_name] # [ model_name, {hyp_name: hyp_val, ...} ]
        hyp_values = {}

    elif line.strip() != "":
        key, value = line.strip().split("\t")
        if "." in value: # Value could either be a float or string
            try:
                hyp_values[key] = float(value)
            except:
                hyp_values[key] = value
        else:
            if value == "None": # Value is a NoneType
                hyp_values[key] = None
            else:
                try: # Value could be an integer or string
                    hyp_values[key] = int(value)
                except:
                    hyp_values[key] = value
    else:
        hyperparams.append(hyp_values)
        all_hyperparams.append(hyperparams)

# Set random seed value
seed = 42
random.seed(seed)

# Set best variable preprocessing combinations and encoding presets (according to results of '2_regression_baseline_results.py' file)
param_comb = [2, 0]
descriptor_preset = [("z-scales", {"n_jobs":2}),
                     ("esm2_3B", {"preset":"features", "pretrained_model":"3B", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2})]

# Start preprocessing combination and encoding preset iterations
for ix, (comb, (preset, params)) in enumerate(zip(*[param_comb, descriptor_preset])):

    best_comb = VARIABLE_PARAMS[int(comb)]

    model_name, model_hyperparameters = all_hyperparams[ix]

    # Recreate Figure 2 from the comparison article (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954)
    make_fig2(model_name, model_hyperparameters, preset, params, best_comb, seed)

    # Recreate Figure 4 from the comparison article (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954).
    # Get the breadth predictions from 'ic50_regression.dat' and 'ic50_regression_approx.dat' files.
    # Save the predictions to recreate figures if needed (time-consuming extraction process)
    with open("breadth_predictions.txt", "a") as f:
        breadth_regression = compute_breadth_all_average(model_name, model_hyperparameters, preset, params, comb, antibodies, 'ic50_regression.dat', 30, seed)
        f.write(f">ic50_regression_{preset}\n{' '.join([str(mean[0]) for mean in breadth_regression.values()])}\n"
                                           f"{' '.join([str(std[1])  for std  in breadth_regression.values()])}\n\n")
        make_fig4(breadth_regression, f'Regressor - {preset}', f'fig4_regressor_{preset}.png')

        breadth_regression_approx = compute_breadth_all_average(model_name, model_hyperparameters, preset, params, comb, antibodies, 'ic50_regression_approx.dat', 30, seed)
        f.write(f">ic50_regression_approx_{preset}\n{' '.join([str(mean[0]) for mean in breadth_regression_approx.values()])}\n"
                                                  f"{' '.join([str(std[1])  for std  in breadth_regression_approx.values()])}\n\n")
        make_fig4(breadth_regression_approx, f'Regressor (w/ approx.) - {preset}', f'fig4_regressor_approx_{preset}.png')

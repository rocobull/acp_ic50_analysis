import sys
sys.path.append("/home/rbullita/Thesis/omnia/omnia-proteins/examples/Pipelines")
# Important to define 'Pipelines' as a source file to function in virtual server

from extra_files import convert_complex_to_encodings, preprocess_steps, pipeline, get_regression_scores,\
                        VARIABLE_PARAMS, GENERAL_PARAMS
import random

# Prepare the random seed value (used in the comparison article - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954)
seed = 42
random.seed(seed)

# Set best variable preprocessing combinations and encoding presets (according to results of '2_regression_baseline_results.py' file)
param_comb = [2, 0]
descriptor_preset = [
                     ("z-scales", {"n_jobs":2}),
                     ("esm2_3B", {"preset":"features", "pretrained_model":"3B", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2})
                    ]

# Start preprocessing combination and encoding preset iterations
for ix, (preset, params) in zip(*[param_comb, descriptor_preset]):

    comb = VARIABLE_PARAMS[ix]

    # Prepare info_to_save and hyperparams_to_save string
    info_to_save = ""
    hyperparams_to_save = ""

    # Get sequence encodings and split data into training and test sets
    X_train, y_train, X_test, y_test = convert_complex_to_encodings("ic50_regression.dat", quantile=0.99,
                                                                    encoder=preset, seed=seed, **params)
    # Get pipeline steps
    steps = preprocess_steps(descriptors=None,
                             **GENERAL_PARAMS,
                             **comb)

    # Run the pipeline function and obtaining best model name and scores
    best_pipeline = pipeline(X_train, y_train,
                             num_trials=1, scheduler="local", searcher="random",
                             problem_type="regression", metric="pearsonr", num_gpus=2,
                             preprocessing_steps=steps,
                             data_name="IC50", comb_num=str(ix), preset_name=preset, folds=False,
                             to_save=True)

    # Get the best model's name
    best_model_name = best_pipeline.best_model

    # If best model name is "WeightedEnsemble_L2", get second best model
    # (can't use this model because the best model must be retrained various times to make figures for performance comparison)
    if best_model_name == "WeightedEnsemble_L2":
        best_model_name = best_pipeline.leaderboard().iloc[1, 0]

    # Get the best (or second best) model
    predictor = best_pipeline.steps[-1][1]
    best_model = predictor.learner._trainer.load_model(best_model_name)

    # Get hyperparameters and save them in the "model_hyperparameters.txt" file (to be retrieved in the '4_regression_test_figures.py' file)
    hyperparams = predictor.learner.info()['model_info'][best_model_name]['hyperparameters']

    with open("model_hyperparameters.txt", "a") as f:
        hyperparams_to_save = f"> {ix}_{preset}_{best_model_name}\t\n"
        for k,v in hyperparams.items():
            hyperparams_to_save += f"{k}\t{v}\n"
        f.write(hyperparams_to_save + "\n")

    # Get model predictions
    X_test, y_test = best_pipeline.transform(X_test, y_test)
    X_test.columns = [str(col) for col in X_test.columns]
    y_pred = best_model.predict(X=X_test)

    # Change y_test to pandas Series (to avoid error in 'get_regression_scores' function)
    y_test = y_test.iloc[:, 0]

    # Get scores obtained from test data predictions and save them to "IC50_scores.txt" file
    scores = get_regression_scores(y_test, y_pred)

    info_to_save += f"> {best_model_name}_{ix}_{preset} - {best_model.parameters}\n"
    info_to_save += "\t".join(["RMSE", "MSE", "r2", "pearsonr", "spearmanr"]) + "\n"
    info_to_save += "\t".join([str(vals) for vals in scores]) + "\n\n"

    with open('IC50_scores.txt', 'a') as f:
        f.write(info_to_save)

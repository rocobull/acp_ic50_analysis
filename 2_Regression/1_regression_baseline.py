import sys
sys.path.append("/home/rbullita/Thesis/omnia/omnia-proteins/examples/Pipelines")
# Important to define 'Pipelines' as a source file to function in virtual server

from extra_files import convert_complex_to_encodings, preprocess_steps, get_regression_scores,\
                        VARIABLE_PARAMS, GENERAL_PARAMS

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import random


# Prepare metric names
METRICS = ["rmse", "mse", "r2", "pearson", "spearman"]

# Prepare the random seed value (used in the comparison article - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006954)
seed = 42
random.seed(seed)

# Prepare encoding preset names and their corresponding hyperparameters
encodings = [("nlf", {"n_jobs":2}),
             ("esm1", {"preset":"representations", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2}),
             ("esm2_8M", {"preset":"representations", "pretrained_model":"8M", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2}),
             ("esm2_35M", {"preset":"representations", "pretrained_model":"35M", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2}),
             ("esm2_150M", {"preset":"representations", "pretrained_model":"150M", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2}),
             ("esm2_650M", {"preset":"features", "pretrained_model":"650M", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2}),
             ("esm2_3B", {"preset":"features", "pretrained_model":"3B", "two_dimensional_embeddings":False, "batch_size":1, "n_jobs":2}),
             ("protbert", {"n_jobs":2}),
             ("z-scales", {"n_jobs":2})]

# Start preprocessing combination and encoding preset iterations
for ix, comb in enumerate(VARIABLE_PARAMS):
    for preset, params in encodings:

        # Prepare info_to_save string and score dictionaries
        info_to_save = ""

        # Get sequence encodings and split data into training and test sets
        X_train, y_train, X_test, y_test = convert_complex_to_encodings("ic50_regression.dat", quantile=0.99,
                                                                        encoder=preset, seed=seed, **params)

        # Get pipeline steps
        steps = preprocess_steps(descriptors=None,
                                 **GENERAL_PARAMS,
                                 **comb)

        # Manually transform data according to the selected preprocessing steps
        for _, transformer in steps:
            transformer.fit(X_train, y_train)
            X_train, y_train = transformer.transform(X_train, y_train)
            X_test, y_test = transformer.transform(X_test, y_test)

        # Change y_test to pandas Series (to avoid error in 'get_regression_scores' function)
        y_test = y_test.iloc[:, 0]

        # Fit baseline models (with hyperparameters used in the comparison article) to training data and get test prediction scores
        svm_predictor = SVR(kernel="rbf")
        rf_predictor = RandomForestRegressor(n_estimators=31)
        knn_predictor = KNeighborsRegressor(n_neighbors=15, weights="distance")

        svm_predictor.fit(X_train, y_train)
        svm_y_pred = svm_predictor.predict(X_test) # [val1, val2, ...]
        svm_metrics = get_regression_scores(y_test, svm_y_pred)

        rf_predictor.fit(X_train, y_train)
        rf_y_pred = rf_predictor.predict(X_test) # [val1, val2, ...]
        rf_metrics = get_regression_scores(y_test, rf_y_pred)

        knn_predictor.fit(X_train, y_train)
        knn_y_pred = knn_predictor.predict(X_test) # [[val1], [val2], ...]
        # Extend values predicted by KNN model
        knn_y_pred = [val[0] for val in knn_y_pred]
        knn_metrics = get_regression_scores(y_test, knn_y_pred)

        # Save scores to a text file
        info_to_save += f"> {ix}_{preset}\n"
        info_to_save += "Model" + "\t" + "\t".join(METRICS) + "\n"
        info_to_save += "SVM" + "\t" + "\t".join([str(vals) for vals in svm_metrics]) + "\n"
        info_to_save += "RF" + "\t" + "\t".join([str(vals) for vals in rf_metrics]) + "\n"
        info_to_save += "KNN" + "\t" + "\t".join([str(vals) for vals in knn_metrics]) + "\n\n"

        with open(f'IC50_baseline_encodings.txt', 'a') as f:
            f.write(info_to_save)

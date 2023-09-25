import sys
sys.path.append("/home/rbullita/Thesis/omnia/omnia-proteins/examples/Pipelines")

from extra_files import convert_acp_to_dataframe, preprocess_steps, x_y_split, get_binary_scores,\
                        VARIABLE_PARAMS, GENERAL_PARAMS, PRESETS

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from omnia.generics import np, pd


# Prepare metric names
METRICS = ["accuracy", "recall", "specificity", "precision", "mcc"]

# Import ACP740 and ACP240 data
data1 = convert_acp_to_dataframe("acp740.txt")
data1.index = range(len(data1.iloc[:, 0]))

data2 = convert_acp_to_dataframe("acp240.txt")
data2.index = range(len(data2.iloc[:, 0]))

# Start data, preprocessing combination and preset iterations
for data, name in zip([data1, data2], ["740", "240"]):

    for ix, comb in enumerate(VARIABLE_PARAMS):
        for preset in PRESETS:

            # Prepare info_to_save string and score dictionaries
            info_to_save = ""

            svm_scores = {} # {"metric_1": [score1, score2, ...], ...}
            rf_scores = {} # {"metric_1": [score1, score2, ...], ...}
            nb_scores = {} # {"metric_1": [score1, score2, ...], ...}

            # Get different train-test data splits for each fold
            for fold in range(5):
                train = pd.DataFrame([data.iloc[i,:] for i,_ in enumerate(data.index) if i % 5 != fold], columns=data.columns)
                X_train, y_train = x_y_split(train, "acp")

                test = pd.DataFrame([data.iloc[i,:] for i,_ in enumerate(data.index) if i % 5 == fold], columns=data.columns)
                X_test, y_test = x_y_split(test, "acp")

                # Get pipeline steps
                steps = preprocess_steps(descriptors=preset,
                                         **GENERAL_PARAMS,
                                         **comb)

                # Manually transform data according to the selected preprocessing steps
                for _,transformer in steps:
                    transformer.fit(X_train, y_train)
                    X_train, y_train = transformer.transform(X_train, y_train)
                    X_test, y_test = transformer.transform(X_test, y_test)

                X_train = X_train.dropna(axis=1)
                X_test = X_test.dropna(axis=1)

                # for column in X_train:
                #     X_train[column] = np.nan_to_num(X_train[column].astype(np.float32))
                #     X_test[column] = np.nan_to_num(X_test[column].astype(np.float32))

                # Fit baseline models to training data split and get test prediction scores
                svm_predictor = SVC()
                rf_predictor = RandomForestClassifier()
                nb_predictor = GaussianNB()

                svm_predictor.fit(X_train, y_train)
                svm_y_pred = svm_predictor.predict(X_test)
                svm_metrics = get_binary_scores(y_test, svm_y_pred)

                rf_predictor.fit(X_train, y_train)
                rf_y_pred = rf_predictor.predict(X_test)
                rf_metrics = get_binary_scores(y_test, rf_y_pred)

                nb_predictor.fit(X_train, y_train)
                nb_y_pred = nb_predictor.predict(X_test)
                nb_metrics = get_binary_scores(y_test, nb_y_pred)

                for i, metric in enumerate(METRICS):
                    svm_scores[metric] = svm_scores.get(metric, []) + [svm_metrics[i]]
                    rf_scores[metric] = rf_scores.get(metric, []) + [rf_metrics[i]]
                    nb_scores[metric] = nb_scores.get(metric, []) + [nb_metrics[i]]

            # Save scores to a text file
            info_to_save += f"> {ix}_{preset}\n"
            info_to_save += "Model" + "\t" + "\t".join(METRICS) + "\n"
            info_to_save += "SVM" + "\t" + "\t".join([str(np.mean(vals)) for vals in svm_scores.values()]) + \
                            "\t(" + "\t".join([str(np.std(vals)) for vals in svm_scores.values()]) + ")\n"
            info_to_save += "RF" + "\t" + "\t".join([str(np.mean(vals)) for vals in rf_scores.values()]) + \
                            "\t(" + "\t".join([str(np.std(vals)) for vals in rf_scores.values()]) + ")\n"
            info_to_save += "NB" + "\t" + "\t".join([str(np.mean(vals)) for vals in nb_scores.values()]) + \
                            "\t(" + "\t".join([str(np.std(vals)) for vals in nb_scores.values()]) + ")\n"
            info_to_save += "\n"

            with open(f'{name}_baseline.txt', 'a') as f:
                f.write(info_to_save)


#predictor = TabularPredictor.load("models/performance\autogluon\")

import sys
sys.path.append("/home/rbullita/Thesis/omnia/omnia-proteins/examples/Pipelines")

from extra_files import convert_acp_to_dataframe, preprocess_steps, pipeline, x_y_split, get_binary_scores,\
                        VARIABLE_PARAMS, GENERAL_PARAMS

from omnia.generics import np, pd


# Import ACP740 and ACP240 data
data1 = convert_acp_to_dataframe("acp740.txt")
data1.index = range(len(data1.iloc[:, 0]))

data2 = convert_acp_to_dataframe("acp240.txt")
data2.index = range(len(data2.iloc[:, 0]))

# Start data, preprocessing combination and descriptor preset iterations
for data, name in zip([data1, data2], ["740", "240"]):

    # Set best variable preprocessing combinations and descriptor presets (according to results of '2_binary_baseline_results.py' file)
    if name == "740":
        param_comb = [3, 1, 2]
        descriptor_preset = ["all", "performance", "all"]
    else:
        param_comb = [1, 5, 3]
        descriptor_preset = ["physico-chemical", "performance", "all"]

    # Prepare dictionary to store predicted probability values from best models
    all_test_proba = {} # {model_id: [sample_pred_1, sample_pred_2, ...], ...}

    for ix, preset in zip(*[param_comb, descriptor_preset]):

        comb = VARIABLE_PARAMS[ix]

        # Prepare info_to_save string and score dictionaries
        info_to_save = ""

        # Getting pipeline steps
        steps = preprocess_steps(descriptors=preset,
                                 **GENERAL_PARAMS,
                                 **comb)

        # Determine the best overall performing model (trained against the training set of the first fold)
        train = pd.DataFrame([data.iloc[i, :] for i, _ in enumerate(data.index) if i % 5 != 0], columns=data.columns)
        X_train, y_train = x_y_split(train, "acp")

        # Run the pipeline function and obtain best model name and scores
        best_pipeline = pipeline(X_train, y_train,
                                   num_trials=100, scheduler="local", searcher="random",
                                   problem_type="binary", metric="accuracy", num_gpus=2,
                                   preprocessing_steps=steps,
                                   data_name=name, comb_num=str(ix), preset_name=preset, folds=False,
                                   to_save=True)

        print(best_pipeline.leaderboard())

        # Get the best model's name
        best_model_name = best_pipeline.leaderboard().iloc[0, 0]

        # If best model name is "WeightedEnsemble_L2" or "MultilayerPerceptronNN", get next best model
        next_ind = 1
        while True:
            if ("WeightedEnsemble" in best_model_name) or ("MultilayerPerceptronNN" in best_model_name):
                best_model_name = best_pipeline.leaderboard().iloc[next_ind, 0]
                next_ind += 1
            else:
                break

        # Get the corresponding model instance
        predictor = best_pipeline.steps[-1][1]
        best_model = predictor.learner._trainer.load_model(best_model_name)

        # print(best_model.params)

        # Save features_to_remove list from the multicollinearity transformer to keep the same preprocessing process
        for step, transformer in steps:
            if step == "multicollinearity":
                features_to_remove = transformer.features_to_remove

        # Get the test set of the first fold
        test = pd.DataFrame([data.iloc[i, :] for i, _ in enumerate(data.index) if i % 5 == 0], columns=data.columns)
        X_test, y_test = x_y_split(test, "acp")

        # Manually transform data according to the selected preprocessing steps
        for step, transformer in steps:
            X_test, y_test = transformer.transform(X_test, y_test)

        # The feature labels saved in the model instance were converted to strings, so it is necessary to do the same to the data feature labels
        X_test.columns = [str(col) for col in X_test.columns]

        # Get model predictions
        y_pred = best_model.predict(X=X_test)

        # Save the scores obtained from the test data predictions
        scores = get_binary_scores(y_test, y_pred)
        all_scores = [scores]

        # Store prediction probabilities
        save_name = f"{best_model_name.split('/')[0]}_{ix}_{preset}"
        all_test_proba[save_name] = all_test_proba.get(save_name, []) + list(best_model.predict_proba(X=X_test))

        # Train and test best model with the remaining 4 folds
        for fold in range(1, 5):
            train = pd.DataFrame([data.iloc[i, :] for i, _ in enumerate(data.index) if i % 5 != fold], columns=data.columns)
            X_train, y_train = x_y_split(train, "acp")

            test = pd.DataFrame([data.iloc[i, :] for i, _ in enumerate(data.index) if i % 5 == fold], columns=data.columns)
            X_test, y_test = x_y_split(test, "acp")

            for step, transformer in steps:
                transformer.fit(X_train, y_train)

                # Manually set the features_to_remove parameter (to keep the same data columns)
                if step == "multicollinearity":
                    transformer.features_to_remove = features_to_remove

                X_train, y_train = transformer.transform(X_train, y_train)
                X_test, y_test = transformer.transform(X_test, y_test)

            # The feature labels saved in the model instance were converted to strings, so it is necessary to do the same to the data feature labels
            X_train.columns = [str(col) for col in X_train.columns]
            X_test.columns = [str(col) for col in X_test.columns]

            # Fit best model
            best_model.fit(X=X_train, y=y_train)

            # Get model predictions
            y_pred = best_model.predict(X=X_test)

            # Append scores obtained from test data predictions
            scores = get_binary_scores(y_test, y_pred)
            all_scores.append(scores)

            # Store prediction probabilities
            all_test_proba[save_name] = all_test_proba.get(save_name, []) + list(best_model.predict_proba(X=X_test))

        # Sort folds according to the "accuracy" value, in descending order
        inds = sorted(range(len(all_scores)), key=lambda i: all_scores[i][0], reverse=True)

        # Save scores to a text file
        info_to_save += f"> {best_model_name}_{ix}_{preset} - {best_model.parameters}\n"
        info_to_save += "Fold" + "\t" + "\t".join(["Accuracy", "Recall", "Specificity", "Precision", "MCC"]) + "\n"
        for i in inds:
            info_to_save += str(i) + "\t" + "\t".join([str(vals) for vals in all_scores[i]]) + "\n"

        # Generate mean and sd values of all metrics for each fold
        mean_sd = [(str(np.mean(vals)), str(np.std(vals)))
                   for vals in zip(*all_scores)]

        info_to_save += "MEAN:" "\t" + "\t".join([mean[0] for mean in mean_sd]) + "\n"
        info_to_save += "SD:  " "\t" + "\t".join([sd[1]   for sd   in mean_sd]) + "\n"
        info_to_save += "\n"

        with open(f'{name}_scores.txt', 'a') as f:
            f.write(info_to_save)

    # Store predicted value probabilities (to be used to later determine ROC curve for each model)
    with open(f'{name}_test_predictions.txt', "a") as f:
        to_write = ""
        for k, v in all_test_proba.items():
            line = "\t".join([str(val) for val in v])
            to_write += f">{k}\n{line}\n\n"
        f.write(to_write)

from typing import Union,List
from .auxiliary_funcs import MODELS

import omnia.generics as omnia
from omnia.generics import pd


def pipeline(X_train:pd.DataFrame, y_train:pd.DataFrame, #X_test:pd.DataFrame, y_test:pd.DataFrame,
             num_trials=10, scheduler="local", searcher="random",
             problem_type="binary", metric="accuracy", models: Union[List[object],str] = MODELS, num_gpus=2,
             preprocessing_steps=None,
             data_name="data", comb_num="0", preset_name="performance", folds=False, fold_num="0",
             to_save=True):
    """
    Parameters
    ----------

    """
    print(f'Running ML analysis with "{preset_name}" descriptors...')

    hyperparameter_tune_kwargs = {
        'num_trials': num_trials,
        'scheduler': scheduler, # ?????
        'searcher': searcher,
    }

    predictor = omnia.TabularPredictor(
        # auto_stack=True,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        problem_type=problem_type,  # ('binary', 'multiclass', 'regression', 'quantile')
        metric=metric, # (accuracy (binary/multiclass), root_mean_squared_error (regression), pinball_loss (quantile))
        models=models, # AUTOMATIC = (best_quality, high_quality, good_quality, medium_quality, optimize_for_deployment, ignore_text)
        num_gpus=num_gpus
    )

    if preprocessing_steps is None:
        preprocessing_steps = []

    # create the pipeline
    if folds:
        path1 = f'models/{data_name}/{comb_num}/{preset_name}/{fold_num}'
    else:
        path1 = f'models/{data_name}/{comb_num}/{preset_name}'


    pipeline = omnia.Pipeline(
        path = path1,
        steps = preprocessing_steps + [('predictor', predictor)]
    )

    # fit the pipeline
    pipeline.fit(X_train, y_train)

    # save the leaderboard and pipeline
    lb = pipeline.leaderboard()

    if folds:
        path2 = f"results/leaderboard_{data_name}_{comb_num}_{preset_name}_{fold_num}.csv"
    else:
        path2 = f"results/leaderboard_{data_name}_{comb_num}_{preset_name}.csv"
    lb.to_csv(path2)

    if to_save:
        pipeline.save()
        predictor.save()

    return pipeline



# def preprocessing_pipeline(X_train: pd.DataFrame, y_train: pd.DataFrame,
#                            X_test: pd.DataFrame = None, y_test: pd.DataFrame = None,
#                            preprocessing_steps=None):
#     """
#     Runs omnia's 'Pipeline' class without a predictor, with the sole purpose of transforming the input data
#     with the given list of preprocessing steps.
#
#     Parameters
#     ----------
#
#     Returns
#     -------
#     The transformed 'X' and 'y' data, and the fitted preprocessing pipeline
#     """
#     if preprocessing_steps is None:
#         preprocessing_steps = []
#
#     pipeline = omnia.Pipeline(steps=preprocessing_steps)
#
#     X_train, y_train = pipeline.fit_transform(X_train, y_train)
#
#     if not (X_test is None):
#         X_test, y_test = pipeline.transform(X_test, y_test)
#
#     return X_train, y_train, X_test, y_test, pipeline
#
#     #fit the pipeline
#     # pipeline.fit(X, y)
#     # new_X, new_y = pipeline.transform(X, y)
#     #
#     # return new_X, new_y, pipeline

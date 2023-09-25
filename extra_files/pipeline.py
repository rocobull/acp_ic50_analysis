from typing import Union, List, Tuple, Type
from .auxiliary_funcs import MODELS

import omnia.generics as omnia
from omnia.generics import Transformer, Model, Pipeline
from omnia.generics import pd


def pipeline(X_train:pd.DataFrame, y_train:pd.DataFrame,
             num_trials:int = 10, scheduler:str = "local", searcher:str = "random",
             problem_type:str = "binary", metric:str = "accuracy",
             models:Union[List[Type[Model]], str] = MODELS, num_gpus:int = 2,
             preprocessing_steps:Union[List[Tuple[str, Type[Transformer]]], None] = None,
             data_name:str = "data", comb_num:str = "0", preset_name:str = "performance",
             folds:bool = False, fold_num:str = "0",
             to_save:bool = True) -> Pipeline:
    """
    A wrapper class that returns a fitted Pipeline class instance (from Omnia).

    Parameters
    ----------
    X_train: pd.DataFrame
        The 'X' training data.
    y_train: pd.DataFrame
        The 'y' training data.
    num_trials: int
        The number of times to train each model type from Omnia.
    scheduler: str
        Method used to control each models' learning rate.
    searcher: str
        Method used for hyperparameter optimization.
    problem_type: str
        The type of ML problem. Can be "binary", "regression", "multiclass" or "quantile".
    metric: str
        The metric used to determine the best performing models.
    models: Union[List[Type[Model]], str]
        A list of model classes from Omnia to be trained against the training data.
        If a str is used, it should be one of the available presets to train a predefined selection of models
        (see Omnia's TabularPredictor class for more information).
    num_gpus: int
        Number of GPUs to use for model training.
    preprocessing_steps: Union[List[Tuple[str, Type[Transformer]]], None]
        A list of tuples, each containing a string value to identify the preprocessing step, and the respective
        preprocessing Transformer.
        If None, the no preprocessing is performed on the data.
    data_name: str
        A name to identify the data being used.
        Used to identify the stored results.
    comb_num: str
        A string value of the variable preprocessing combination index used.
        Used to identify the stored results.
    preset_name: str
        The name of the descriptor/encoding preset used.
        Used to identify the stored results.
    folds: bool
        A boolean value indicating if folds were used to generate the training data (True) or not (False).
    fold_num: str
        A string value of the n^th fold used (if 'folds' = True).
        Used to identify the stored results.
    to_save: bool
        Indicates whether to save the pipeline and predictor instances (True) or not (False)

    Returns
    -------
    pipeline_inst: Pipeline
        A Pipeline instance fitted to the given training data.
    """
    print(f'Running ML analysis with "{preset_name}" descriptors...')

    # Create the TabularPredictor instance
    hyperparameter_tune_kwargs = {
        'num_trials': num_trials,
        'scheduler': scheduler,
        'searcher': searcher,
    }
    predictor = omnia.TabularPredictor(
        # auto_stack=True,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        problem_type=problem_type,  # ('binary', 'multiclass', 'regression', 'quantile')
        metric=metric, # ("accuracy" (binary/multiclass), "root_mean_squared_error" (regression), "pinball_loss" (quantile))
        models=models, # Presets: best_quality, high_quality, good_quality, medium_quality, optimize_for_deployment, ignore_text
        num_gpus=num_gpus
    )

    # Define empty list if preprocessing_steps is None
    if preprocessing_steps is None:
        preprocessing_steps = []

    # Create the Pipeline instance from Omnia
    if folds:
        path1 = f'models/{data_name}/{comb_num}/{preset_name}/{fold_num}'
    else:
        path1 = f'models/{data_name}/{comb_num}/{preset_name}'

    pipeline_inst = Pipeline(
        path = path1,
        steps = preprocessing_steps + [('predictor', predictor)]
    )

    # Fit the pipeline
    pipeline_inst.fit(X_train, y_train)

    # Save the leaderboard, pipeline and predictor
    lb = pipeline_inst.leaderboard()

    if folds:
        path2 = f"results/leaderboard_{data_name}_{comb_num}_{preset_name}_{fold_num}.csv"
    else:
        path2 = f"results/leaderboard_{data_name}_{comb_num}_{preset_name}.csv"
    lb.to_csv(path2)

    if to_save:
        pipeline_inst.save()
        predictor.save()

    return pipeline_inst


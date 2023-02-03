import warnings

import xgboost as xgb
from config.core import config
from pipeline import full_processor as fp
from pipeline import preprocessing_pipeline as pp
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    y_train = y_train

    model = xgb.XGBClassifier()

    xgbpipeline = Pipeline(
        steps=[("preprocess", pp), ("transform", fp), ("model", model)]
    )

    # fit model
    params = {"model__learning_rate": [0.2, 0.4, 0.7]}

    search = GridSearchCV(xgbpipeline, params, cv=2, scoring="precision")

    search.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=search)


if __name__ == "__main__":
    run_training()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from garcon_model.config.core import config
from garcon_model.processing import features as pp

preprocessing_pipeline = Pipeline(
    steps=[
        ("season_transformation", pp.SeasonTransformer()),
        ("price_transformation", pp.PriceTransformer()),
    ]
)

numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
)

categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

date_pipeline = Pipeline(
    steps=[("datecategorizer", OrdinalEncoder(handle_unknown="error"))]
)

full_processor = ColumnTransformer(
    transformers=[
        ("number", numeric_pipeline, config.model_config.numerical_vars),
        ("category", categorical_pipeline, config.model_config.categorical_vars),
        ("dates", date_pipeline, config.model_config.temporal_vars),
    ]
)

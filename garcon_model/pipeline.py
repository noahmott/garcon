from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Binarizer, OrdinalEncoder

from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, CategoricalImputer
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from garcon_model.config.core import config
from garcon_model.processing import features as pp
import xgboost as xgb

preprocessing_pipeline=Pipeline(steps=[
    ('season_transformation', pp.SeasonTransformer()),
    ('price_transformation', pp.PriceTransformer())
])

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
])

date_pipeline=Pipeline(steps=[
    ('datecategorizer', OrdinalEncoder(handle_unknown='error'))
])

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, config.model_config.numerical_vars),
    ('category', categorical_pipeline, config.model_config.categorical_vars),
    ('dates', date_pipeline, config.model_config.temporal_vars)
])




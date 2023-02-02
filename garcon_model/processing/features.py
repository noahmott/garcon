from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SeasonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit( self, X, y = None ):
        return self

    def transform(self, X, y=None):
      def season(x):
        if x in [9,10,11]:
            return 'Autumn'
        if x in [1,2,12]:
            return 'Winter'
        if x in [3,4,5]:
            return 'Spring'
        if x in [6,7,8]:
            return 'Summer'
        return x
      X['season']=X['arrival_month'].apply(season)
      return (X)

class PriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit( self, X, y = None ):
        return self

    def transform(self, X, y=None):
      def avg_price_per_room_group(x):
        if x <= 50.0 :
            x= 'price below 50'
        elif x >50.0 and x <=150.0:
            x= 'price from 50 to 150'
        elif x >150.0 and x <=300.0:
            x= 'price from 150 to 300'
        elif x >300.0 and x <=450.0:
            x= 'price from 300 to 450'
        else:
            x= 'price 450+'
        return x
      X['avg_price_per_room_group']=X['avg_price_per_room'].apply(avg_price_per_room_group)
      return (X)

'''
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Temporal elapsed time transformer."""

    def __init__(self, variables: List[str], reference_variable: str):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: List[str], mappings: dict):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X
'''
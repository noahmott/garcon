from sklearn.base import BaseEstimator, TransformerMixin


class SeasonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def season(x):
            if x in [9, 10, 11]:
                return "Autumn"
            if x in [1, 2, 12]:
                return "Winter"
            if x in [3, 4, 5]:
                return "Spring"
            if x in [6, 7, 8]:
                return "Summer"
            return x

        X["season"] = X["arrival_month"].apply(season)
        return X


class PriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def avg_price_per_room_group(x):
            if x <= 50.0:
                x = "price below 50"
            elif x > 50.0 and x <= 150.0:
                x = "price from 50 to 150"
            elif x > 150.0 and x <= 300.0:
                x = "price from 150 to 300"
            elif x > 300.0 and x <= 450.0:
                x = "price from 300 to 450"
            else:
                x = "price 450+"
            return x

        X["avg_price_per_room_group"] = X["avg_price_per_room"].apply(
            avg_price_per_room_group
        )
        return X

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from garcon_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.categorical_vars_with_na_frequent
        + config.model_config.categorical_vars_with_na_missing
        + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data

'''
def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    input_data["MSSubClass"] = input_data["MSSubClass"].astype("O")
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleHouseDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

'''
class GarconInputSchema(BaseModel):
    type_of_meal_plan: Optional[str]
    no_of_adults: Optional[int]
    room_type_reserved: Optional[str]
    market_segment_type: Optional[str]
    no_of_children: Optional[float]
    no_of_weekend_nights: Optional[float]
    no_of_week_nights: Optional[float]
    required_car_parking_space: Optional[float]
    lead_time: Optional[float]
    arrival_month: Optional[int]
    arrival_date: Optional[int]
    repeated_guest: Optional[int]
    avg_price_per_room: Optional[float]
    no_of_previous_cancellations: Optional[int]
    no_of_previous_bookings_not_canceled: Optional[int]
    no_of_special_requests: Optional[int]
    canceled: Optional[int]
    season: Optional[str]
    avg_price_per_room_group: Optional[str]
    arrival_year: Optional[int]

class GarconDataInputs(BaseModel):
    inputs: List[GarconInputSchema]

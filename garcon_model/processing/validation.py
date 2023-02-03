from typing import List, Optional, Tuple
from pydantic import BaseModel, ValidationError
import pandas as pd
from garcon_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data

def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    # input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    # input_data["MSSubClass"] = input_data["MSSubClass"].astype("O")
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    return validated_data, errors


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

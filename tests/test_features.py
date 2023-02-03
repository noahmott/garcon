from sklearn.preprocessing import OrdinalEncoder
import pytest
from garcon_model.config.core import config

@pytest.mark.skip(reason='testing not setup')
def test_date_transformer(sample_input_data):
    # Given
    transformer=OrdinalEncoder()

    transformer.fit_transform(sample_input_data["arrival_year"])
    
    transformer
    #print(type(subject))



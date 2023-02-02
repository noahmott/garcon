import pytest

from garcon_model.config.core import config
from garcon_model.processing.data_manager import load_dataset
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.test_data_file)

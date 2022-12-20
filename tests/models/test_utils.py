import pytest

from ddpg.models.utils import load_model, save_model
from ddpg.models.errors import ClassTypeNotAllowed


class MockedModel(object):
    pass


class TestLoadMethod(object):
    def test_raisesClassTypeNotAllowed(self):
        with pytest.raises(ClassTypeNotAllowed):
            load_model("mocked_path_not_exist", model_class=MockedModel)


class TestSaveMethod(object):
    def test_raisesClassTypeNotAllowed(self):
        with pytest.raises(ClassTypeNotAllowed):
            save_model(model=MockedModel(), filepath="mocked_path_not_exist")

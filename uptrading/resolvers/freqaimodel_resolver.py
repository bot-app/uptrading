# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for upai
"""
import logging
from pathlib import Path

from uptrading.constants import USERPATH_UPAIMODELS, Config
from uptrading.exceptions import OperationalException
from uptrading.upai.upai_interface import IUpaiModel
from uptrading.resolvers import IResolver


logger = logging.getLogger(__name__)


class UpaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = IUpaiModel
    object_type_str = "UpaiModel"
    user_subdir = USERPATH_UPAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("upai/prediction_models").resolve()
    )
    extra_path = "upaimodel_path"

    @staticmethod
    def load_upaimodel(config: Config) -> IUpaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        upaimodel_name = config.get("upaimodel")
        if not upaimodel_name:
            raise OperationalException(
                "No upaimodel set. Please use `--upaimodel` to "
                "specify the UpaiModel class to use.\n"
            )
        if upaimodel_name in disallowed_models:
            raise OperationalException(
                f"{upaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        upaimodel = UpaiModelResolver.load_object(
            upaimodel_name,
            config,
            kwargs={"config": config},
        )

        return upaimodel

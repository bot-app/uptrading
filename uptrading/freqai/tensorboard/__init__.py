# ensure users can still use a non-torch upai version
try:
    from uptrading.upai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger
    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from uptrading.upai.tensorboard.base_tensorboard import (BaseTensorBoardCallback,
                                                               BaseTensorboardLogger)
    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = (
    "TBLogger",
    "TBCallback"
)

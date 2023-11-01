import logging
import platform
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from uptrading.configuration import TimeRange
from uptrading.data.dataprovider import DataProvider
from uptrading.enums import RunMode
from uptrading.upai.data_kitchen import UpaiDataKitchen
from uptrading.upai.utils import download_all_data_for_training, get_required_data_timerange
from uptrading.optimize.backtesting import Backtesting
from uptrading.persistence import Trade
from uptrading.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, log_has_re
from tests.upai.conftest import (get_patched_upai_strategy, is_mac, make_rl_config,
                                   mock_pytorch_mlp_model_training_parameters)


def is_py11() -> bool:
    return sys.version_info >= (3, 11)


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


def can_run_model(model: str) -> None:
    if is_arm() and "Catboost" in model:
        pytest.skip("CatBoost is not supported on ARM.")

    is_pytorch_model = 'Reinforcement' in model or 'PyTorch' in model
    if is_pytorch_model and is_mac() and not is_arm():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize('model, pca, dbscan, float32, can_short, shuffle, buffer, noise', [
    ('LightGBMRegressor', True, False, True, True, False, 0, 0),
    ('XGBoostRegressor', False, True, False, True, False, 10, 0.05),
    ('XGBoostRFRegressor', False, False, False, True, False, 0, 0),
    ('CatboostRegressor', False, False, False, True, True, 0, 0),
    ('PyTorchMLPRegressor', False, False, False, False, False, 0, 0),
    ('PyTorchTransformerRegressor', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner', False, True, False, True, False, 0, 0),
    ('ReinforcementLearner_multiproc', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, False, False, 0, 0),
    ('ReinforcementLearner_test_3ac', False, False, False, True, False, 0, 0),
    ('ReinforcementLearner_test_4ac', False, False, False, True, False, 0, 0),
    ])
def test_extract_data_and_train_model_Standard(mocker, upai_conf, model, pca,
                                               dbscan, float32, can_short, shuffle,
                                               buffer, noise):

    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = 'joblib'
    upai_conf.update({"upaimodel": model})
    upai_conf.update({"timerange": "20180110-20180130"})
    upai_conf.update({"strategy": "upai_test_strat"})
    upai_conf['upai']['feature_parameters'].update({"principal_component_analysis": pca})
    upai_conf['upai']['feature_parameters'].update({"use_DBSCAN_to_remove_outliers": dbscan})
    upai_conf.update({"reduce_df_footprint": float32})
    upai_conf['upai']['feature_parameters'].update({"shuffle_after_split": shuffle})
    upai_conf['upai']['feature_parameters'].update({"buffer_train_data_candles": buffer})
    upai_conf['upai']['feature_parameters'].update({"noise_standard_deviation": noise})

    if 'ReinforcementLearner' in model:
        model_save_ext = 'zip'
        upai_conf = make_rl_config(upai_conf)
        # test the RL guardrails
        upai_conf['upai']['feature_parameters'].update({"use_SVM_to_remove_outliers": True})
        upai_conf['upai']['feature_parameters'].update({"DI_threshold": 2})
        upai_conf['upai']['data_split_parameters'].update({'shuffle': True})

    if 'test_3ac' in model or 'test_4ac' in model:
        upai_conf["upaimodel_path"] = str(Path(__file__).parents[1] / "upai" / "test_models")
        upai_conf["upai"]["rl_config"]["drop_ohlc_from_features"] = True

    if 'PyTorch' in model:
        model_save_ext = 'zip'
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        upai_conf['upai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            upai_conf.update({"conv_width": 10})

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.activate_tensorboard = test_tb
    upai.can_short = can_short
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    upai.dk.set_paths('ADA/BTC', 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)

    upai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    upai.dk.set_paths('ADA/BTC', None)

    upai.train_timer("start", "ADA/BTC")
    upai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, upai.dk, data_load_timerange)
    upai.train_timer("stop", "ADA/BTC")
    upai.dd.save_metric_tracker_to_disk()
    upai.dd.save_drawer_to_disk()

    assert Path(upai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(upai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(upai.dk.data_path /
                f"{upai.dk.model_filename}_model.{model_save_ext}").is_file()
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_metadata.json").is_file()
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(upai.dk.full_path))


@pytest.mark.parametrize('model, strat', [
    ('LightGBMRegressorMultiTarget', "upai_test_multimodel_strat"),
    ('XGBoostRegressorMultiTarget', "upai_test_multimodel_strat"),
    ('CatboostRegressorMultiTarget', "upai_test_multimodel_strat"),
    ('LightGBMClassifierMultiTarget', "upai_test_multimodel_classifier_strat"),
    ('CatboostClassifierMultiTarget', "upai_test_multimodel_classifier_strat")
    ])
def test_extract_data_and_train_model_MultiTargets(mocker, upai_conf, model, strat):
    can_run_model(model)

    upai_conf.update({"timerange": "20180110-20180130"})
    upai_conf.update({"strategy": strat})
    upai_conf.update({"upaimodel": model})
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)

    upai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    upai.dk.set_paths('ADA/BTC', None)

    upai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, upai.dk, data_load_timerange)

    assert len(upai.dk.label_list) == 2
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_model.joblib").is_file()
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_metadata.json").is_file()
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(upai.dk.data['training_features_list']) == 14

    shutil.rmtree(Path(upai.dk.full_path))


@pytest.mark.parametrize('model', [
    'LightGBMClassifier',
    'CatboostClassifier',
    'XGBoostClassifier',
    'XGBoostRFClassifier',
    'PyTorchMLPClassifier',
    ])
def test_extract_data_and_train_model_Classifiers(mocker, upai_conf, model):
    can_run_model(model)

    upai_conf.update({"upaimodel": model})
    upai_conf.update({"strategy": "upai_test_classifier"})
    upai_conf.update({"timerange": "20180110-20180130"})
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)

    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)

    upai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    upai.dk.set_paths('ADA/BTC', None)

    upai.extract_data_and_train_model(new_timerange, "ADA/BTC",
                                        strategy, upai.dk, data_load_timerange)

    if 'PyTorchMLPClassifier':
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        upai_conf['upai']['model_training_parameters'].update(pytorch_mlp_mtp)

    if upai.dd.model_type == 'joblib':
        model_file_extension = ".joblib"
    elif upai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(f"Unsupported model type: {upai.dd.model_type},"
                        f" can't assign model_file_extension")

    assert Path(upai.dk.data_path /
                f"{upai.dk.model_filename}_model{model_file_extension}").exists()
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_metadata.json").exists()
    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(upai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "upai_test_strat"),
        ("XGBoostRegressor", 2, "upai_test_strat"),
        ("CatboostRegressor", 2, "upai_test_strat"),
        ("PyTorchMLPRegressor", 2, "upai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "upai_test_strat"),
        ("ReinforcementLearner", 3, "upai_rl_test_strat"),
        ("XGBoostClassifier", 2, "upai_test_classifier"),
        ("LightGBMClassifier", 2, "upai_test_classifier"),
        ("CatboostClassifier", 2, "upai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "upai_test_classifier")
    ],
    )
def test_start_backtesting(mocker, upai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac():
        test_tb = False

    upai_conf.get("upai", {}).update({"save_backtest_models": True})
    upai_conf['runmode'] = RunMode.BACKTEST

    Trade.use_db = False

    upai_conf.update({"upaimodel": model})
    upai_conf.update({"timerange": "20180120-20180130"})
    upai_conf.update({"strategy": strat})

    if 'ReinforcementLearner' in model:
        upai_conf = make_rl_config(upai_conf)

    if 'test_4ac' in model:
        upai_conf["upaimodel_path"] = str(Path(__file__).parents[1] / "upai" / "test_models")

    if 'PyTorch' in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        upai_conf['upai']['model_training_parameters'].update(pytorch_mlp_mtp)
        if 'Transformer' in model:
            # transformer model takes a window, unlike the MLP regressor
            upai_conf.update({"conv_width": 10})

    upai_conf.get("upai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = False
    upai.activate_tensorboard = test_tb
    upai.dk = UpaiDataKitchen(upai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)
    df = base_df[upai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    upai.dk.set_paths('LTC/BTC', None)
    upai.start_backtesting(df, metadata, upai.dk, strategy)
    model_folders = [x for x in upai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(upai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, upai_conf):
    upai_conf.update({"timerange": "20180120-20180124"})
    upai_conf.get("upai", {}).update({"backtest_period_days": 0.5})
    upai_conf.get("upai", {}).update({"save_backtest_models": True})
    upai_conf.get("upai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)
    df = base_df[upai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    upai.start_backtesting(df, metadata, upai.dk, strategy)
    model_folders = [x for x in upai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(upai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, upai_conf, caplog):
    upai_conf.update({"timerange": "20180120-20180130"})
    upai_conf.get("upai", {}).update({"save_backtest_models": True})
    upai_conf.get("upai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]})
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)
    df = base_df[upai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    upai.dk.pair = pair
    upai.start_backtesting(df, metadata, upai.dk, strategy)
    model_folders = [x for x in upai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

    upai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)
    df = base_df[upai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    upai.dk.pair = pair
    upai.start_backtesting(df, metadata, upai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    upai.dk.pair = pair
    upai.start_backtesting(df, metadata, upai.dk, strategy)

    path = (upai.dd.full_path / upai.dk.backtest_predictions_folder)
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(upai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, upai_conf, caplog):
    upai_conf.get("upai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)
    df = upai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_upai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = upai.dk.remove_special_chars_from_feature_names(df)
    upai.dk.get_unique_classes_from_labels(df)
    upai.dk.pair = "ADA/BTC"
    upai.dk.full_df = df.fillna(0)
    upai.dk.full_df
    assert "&-s_close_mean" not in upai.dk.full_df.columns
    assert "&-s_close_std" not in upai.dk.full_df.columns
    upai.backtesting_fit_live_predictions(upai.dk)
    assert "&-s_close_mean" in upai.dk.full_df.columns
    assert "&-s_close_std" in upai.dk.full_df.columns
    shutil.rmtree(Path(upai.dk.full_path))


def test_plot_feature_importance(mocker, upai_conf):

    from uptrading.upai.utils import plot_feature_importance

    upai_conf.update({"timerange": "20180110-20180130"})
    upai_conf.get("upai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"})

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)

    upai.dd.pair_dict = {"ADA/BTC": {"model_filename": "fake_name",
                                       "trained_timestamp": 1, "data_path": "", "extras": {}}}

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    upai.dk.set_paths('ADA/BTC', None)

    upai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, upai.dk, data_load_timerange)

    model = upai.dd.load_data("ADA/BTC", upai.dk)

    plot_feature_importance(model, "ADA/BTC", upai.dk)

    assert Path(upai.dk.data_path / f"{upai.dk.model_filename}.html")

    shutil.rmtree(Path(upai.dk.full_path))


@pytest.mark.parametrize('timeframes,corr_pairs', [
    (['5m'], ['ADA/BTC', 'DASH/BTC']),
    (['5m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
    (['5m', '15m'], ['ADA/BTC', 'DASH/BTC', 'ETH/USDT']),
])
def test_upai_informative_pairs(mocker, upai_conf, timeframes, corr_pairs):
    upai_conf['upai']['feature_parameters'].update({
        'include_timeframes': timeframes,
        'include_corr_pairlist': corr_pairs,

    })
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    pairlists = PairListManager(exchange, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)


def test_start_set_train_queue(mocker, upai_conf, caplog):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    pairlist = PairListManager(exchange, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange, pairlist)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = False

    upai.train_queue = upai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, upai_conf):
    time_range = get_required_data_timerange(upai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, upai_conf, caplog, tmpdir):
    caplog.set_level(logging.DEBUG)
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    pairlist = PairListManager(exchange, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange, pairlist)
    upai_conf['pairs'] = upai_conf['exchange']['pair_whitelist']
    upai_conf['datadir'] = Path(tmpdir)
    download_all_data_for_training(strategy.dp, upai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize('dp_exists', [(False), (True)])
def test_get_state_info(mocker, upai_conf, dp_exists, caplog, tickers):

    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")
    if is_py11():
        pytest.skip("Reinforcement learning currently not available on python 3.11.")

    upai_conf.update({"upaimodel": "ReinforcementLearner"})
    upai_conf.update({"timerange": "20180110-20180130"})
    upai_conf.update({"strategy": "upai_rl_test_strat"})
    upai_conf = make_rl_config(upai_conf)
    upai_conf['entry_pricing']['price_side'] = 'same'
    upai_conf['exit_pricing']['price_side'] = 'same'

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    ticker_mock = MagicMock(return_value=tickers()['ETH/BTC'])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(upai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.data_provider = strategy.dp
    upai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    upai.get_state_info("ADA/BTC")
    upai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )

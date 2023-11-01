import platform
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from uptrading.configuration import TimeRange
from uptrading.data.dataprovider import DataProvider
from uptrading.upai.data_drawer import UpaiDataDrawer
from uptrading.upai.data_kitchen import UpaiDataKitchen
from uptrading.resolvers import StrategyResolver
from uptrading.resolvers.upaimodel_resolver import UpaiModelResolver
from tests.conftest import get_patched_exchange


def is_mac() -> bool:
    machine = platform.system()
    return "Darwin" in machine


@pytest.fixture(scope="function")
def upai_conf(default_conf, tmpdir):
    upaiconf = deepcopy(default_conf)
    upaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "strategy": "upai_test_strat",
            "user_data_dir": Path(tmpdir),
            "strategy-path": "uptrading/tests/strategy/strats",
            "upaimodel": "LightGBMRegressor",
            "upaimodel_path": "upai/prediction_models",
            "timerange": "20180110-20180115",
            "upai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "uniqe-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path('config_examples', 'config_upai.example.json')]
        }
    )
    upaiconf['exchange'].update({'pair_whitelist': ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC']})
    return upaiconf


def make_rl_config(conf):
    conf.update({"strategy": "upai_rl_test_strat"})
    conf["upai"].update({"model_training_parameters": {
        "learning_rate": 0.00025,
        "gamma": 0.9,
        "verbose": 1
    }})
    conf["upai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {
            "rr": 1,
            "profit_aim": 0.02,
            "win_reward_factor": 2
        },
        "drop_ohlc_from_features": False
        }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> Dict[str, Any]:
    return {
            "learning_rate": 3e-4,
            "trainer_kwargs": {
                "n_steps": None,
                "batch_size": 64,
                "n_epochs": 1,
            },
            "model_kwargs": {
                "hidden_dim": 32,
                "dropout_percent": 0.2,
                "n_layer": 1,
            }
        }


def get_patched_data_kitchen(mocker, upaiconf):
    dk = UpaiDataKitchen(upaiconf)
    return dk


def get_patched_data_drawer(mocker, upaiconf):
    # dd = mocker.patch('uptrading.upai.data_drawer', MagicMock())
    dd = UpaiDataDrawer(upaiconf)
    return dd


def get_patched_upai_strategy(mocker, upaiconf):
    strategy = StrategyResolver.load_strategy(upaiconf)
    strategy.ft_bot_start()

    return strategy


def get_patched_upaimodel(mocker, upaiconf):
    upaimodel = UpaiModelResolver.load_upaimodel(upaiconf)

    return upaimodel


def make_unfiltered_dataframe(mocker, upai_conf):
    upai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    upai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(data_load_timerange, upai.dk)

    upai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = upai.dd.get_base_and_corr_dataframes(
            data_load_timerange, upai.dk.pair, upai.dk
        )

    unfiltered_dataframe = upai.dk.use_strategy_to_populate_indicators(
                strategy, corr_dataframes, base_dataframes, upai.dk.pair
            )
    for i in range(5):
        unfiltered_dataframe[f'constant_{i}'] = i

    unfiltered_dataframe = upai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return upai, unfiltered_dataframe


def make_data_dictionary(mocker, upai_conf):
    upai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    upai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    upai.dd.load_all_pair_histories(data_load_timerange, upai.dk)

    upai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = upai.dd.get_base_and_corr_dataframes(
            data_load_timerange, upai.dk.pair, upai.dk
        )

    unfiltered_dataframe = upai.dk.use_strategy_to_populate_indicators(
                strategy, corr_dataframes, base_dataframes, upai.dk.pair
            )

    unfiltered_dataframe = upai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    upai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = upai.dk.filter_features(
            unfiltered_dataframe,
            upai.dk.training_features_list,
            upai.dk.label_list,
            training_filter=True,
        )

    data_dictionary = upai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = upai.dk.normalize_data(data_dictionary)

    return upai


def get_upai_live_analyzed_dataframe(mocker, upaiconf):
    strategy = get_patched_upai_strategy(mocker, upaiconf)
    exchange = get_patched_exchange(mocker, upaiconf)
    strategy.dp = DataProvider(upaiconf, exchange)
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upaiconf, upai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    upai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair('ADA/BTC', '5m')
    return strategy.dp.get_analyzed_dataframe('ADA/BTC', '5m')


def get_upai_analyzed_dataframe(mocker, upaiconf):
    strategy = get_patched_upai_strategy(mocker, upaiconf)
    exchange = get_patched_exchange(mocker, upaiconf)
    strategy.dp = DataProvider(upaiconf, exchange)
    strategy.upai_info = upaiconf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upaiconf, upai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    upai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = upai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return upai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')


def get_ready_to_train(mocker, upaiconf):
    strategy = get_patched_upai_strategy(mocker, upaiconf)
    exchange = get_patched_exchange(mocker, upaiconf)
    strategy.dp = DataProvider(upaiconf, exchange)
    strategy.upai_info = upaiconf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upaiconf, upai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    upai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = upai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, upai, strategy

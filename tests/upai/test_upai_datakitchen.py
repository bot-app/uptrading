import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from uptrading.configuration import TimeRange
from uptrading.data.dataprovider import DataProvider
from uptrading.exceptions import OperationalException
from uptrading.upai.data_kitchen import UpaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.upai.conftest import (get_patched_data_kitchen, get_patched_upai_strategy,
                                   make_unfiltered_dataframe)
from tests.upai.test_upai_interface import is_mac


@pytest.mark.parametrize(
    "timerange, train_period_days, expected_result",
    [
        ("20220101-20220201", 30, "20211202-20220201"),
        ("20220301-20220401", 15, "20220214-20220401"),
    ],
)
def test_create_fulltimerange(
    timerange, train_period_days, expected_result, upai_conf, mocker, caplog
):
    dk = get_patched_data_kitchen(mocker, upai_conf)
    assert dk.create_fulltimerange(timerange, train_period_days) == expected_result
    shutil.rmtree(Path(dk.full_path))


def test_create_fulltimerange_incorrect_backtest_period(mocker, upai_conf):
    dk = get_patched_data_kitchen(mocker, upai_conf)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be an integer"):
        dk.create_fulltimerange("20220101-20220201", 0.5)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be positive"):
        dk.create_fulltimerange("20220101-20220201", -1)
    shutil.rmtree(Path(dk.full_path))


@pytest.mark.parametrize(
    "timerange, train_period_days, backtest_period_days, expected_result",
    [
        ("20220101-20220201", 30, 7, 9),
        ("20220101-20220201", 30, 0.5, 120),
        ("20220101-20220201", 10, 1, 80),
    ],
)
def test_split_timerange(
    mocker, upai_conf, timerange, train_period_days, backtest_period_days, expected_result
):
    upai_conf.update({"timerange": "20220101-20220401"})
    dk = get_patched_data_kitchen(mocker, upai_conf)
    tr_list, bt_list = dk.split_timerange(timerange, train_period_days, backtest_period_days)
    assert len(tr_list) == len(bt_list) == expected_result

    with pytest.raises(
        OperationalException, match=r"train_period_days must be an integer greater than 0."
    ):
        dk.split_timerange("20220101-20220201", -1, 0.5)
    shutil.rmtree(Path(dk.full_path))


def test_check_if_model_expired(mocker, upai_conf):

    dk = get_patched_data_kitchen(mocker, upai_conf)
    now = datetime.now(tz=timezone.utc).timestamp()
    assert dk.check_if_model_expired(now) is False
    now = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).timestamp()
    assert dk.check_if_model_expired(now) is True
    shutil.rmtree(Path(dk.full_path))


def test_filter_features(mocker, upai_conf):
    upai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, upai_conf)
    upai.dk.find_features(unfiltered_dataframe)

    filtered_df, labels = upai.dk.filter_features(
            unfiltered_dataframe,
            upai.dk.training_features_list,
            upai.dk.label_list,
            training_filter=True,
    )

    assert len(filtered_df.columns) == 14


def test_make_train_test_datasets(mocker, upai_conf):
    upai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, upai_conf)
    upai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = upai.dk.filter_features(
            unfiltered_dataframe,
            upai.dk.training_features_list,
            upai.dk.label_list,
            training_filter=True,
        )

    data_dictionary = upai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    assert data_dictionary
    assert len(data_dictionary) == 7
    assert len(data_dictionary['train_features'].index) == 1916


@pytest.mark.parametrize('model', [
    'LightGBMRegressor'
    ])
def test_get_full_model_path(mocker, upai_conf, model):
    upai_conf.update({"upaimodel": model})
    upai_conf.update({"timerange": "20180110-20180130"})
    upai_conf.update({"strategy": "upai_test_strat"})

    if is_mac():
        pytest.skip("Mac is confused during this test for unknown reasons")

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

    model_path = upai.dk.get_full_models_path(upai_conf)
    assert model_path.is_dir() is True

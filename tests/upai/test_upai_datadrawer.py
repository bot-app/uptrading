
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from uptrading.configuration import TimeRange
from uptrading.data.dataprovider import DataProvider
from uptrading.exceptions import OperationalException
from uptrading.upai.data_kitchen import UpaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.upai.conftest import get_patched_upai_strategy


def test_update_historic_data(mocker, upai_conf):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")

    upai.dd.load_all_pair_histories(timerange, upai.dk)
    historic_candles = len(upai.dd.historic_data["ADA/BTC"]["5m"])
    dp_candles = len(strategy.dp.get_pair_dataframe("ADA/BTC", "5m"))
    candle_difference = dp_candles - historic_candles
    upai.dk.pair = "ADA/BTC"
    upai.dd.update_historic_data(strategy, upai.dk)

    updated_historic_candles = len(upai.dd.historic_data["ADA/BTC"]["5m"])

    assert updated_historic_candles - historic_candles == candle_difference
    shutil.rmtree(Path(upai.dk.full_path))


def test_load_all_pairs_histories(mocker, upai_conf):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    upai.dd.load_all_pair_histories(timerange, upai.dk)

    assert len(upai.dd.historic_data.keys()) == len(
        upai_conf.get("exchange", {}).get("pair_whitelist")
    )
    assert len(upai.dd.historic_data["ADA/BTC"]) == len(
        upai_conf.get("upai", {}).get("feature_parameters", {}).get("include_timeframes")
    )
    shutil.rmtree(Path(upai.dk.full_path))


def test_get_base_and_corr_dataframes(mocker, upai_conf):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)

    num_tfs = len(
        upai_conf.get("upai", {}).get("feature_parameters", {}).get("include_timeframes")
    )

    assert len(base_df.keys()) == num_tfs

    assert len(corr_df.keys()) == len(
        upai_conf.get("upai", {}).get("feature_parameters", {}).get("include_corr_pairlist")
    )

    assert len(corr_df["ADA/BTC"].keys()) == num_tfs
    shutil.rmtree(Path(upai.dk.full_path))


def test_use_strategy_to_populate_indicators(mocker, upai_conf):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    strategy.upai_info = upai_conf.get("upai", {})
    upai = strategy.upai
    upai.live = True
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180114")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", upai.dk)

    df = upai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, 'LTC/BTC')

    assert len(df.columns) == 33
    shutil.rmtree(Path(upai.dk.full_path))


def test_get_timerange_from_live_historic_predictions(mocker, upai_conf):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    upai.dk.live = False
    timerange = TimeRange.parse_timerange("20180126-20180130")
    upai.dd.load_all_pair_histories(timerange, upai.dk)
    sub_timerange = TimeRange.parse_timerange("20180128-20180130")
    _, base_df = upai.dd.get_base_and_corr_dataframes(sub_timerange, "ADA/BTC", upai.dk)
    base_df["5m"]["date_pred"] = base_df["5m"]["date"]
    upai.dd.historic_predictions = {}
    upai.dd.historic_predictions["ADA/USDT"] = base_df["5m"]
    upai.dd.save_historic_predictions_to_disk()
    upai.dd.save_global_metadata_to_disk({"start_dry_live_date": 1516406400})

    timerange = upai.dd.get_timerange_from_live_historic_predictions()
    assert timerange.startts == 1516406400
    assert timerange.stopts == 1517356500


def test_get_timerange_from_backtesting_live_df_pred_not_found(mocker, upai_conf):
    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    with pytest.raises(
            OperationalException,
            match=r'Historic predictions not found.*'
            ):
        upai.dd.get_timerange_from_live_historic_predictions()


def test_set_initial_return_values(mocker, upai_conf):
    """
    Simple test of the set initial return values that ensures
    we are concatening and ffilling values properly.
    """

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    # Setup
    pair = "BTC/USD"
    end_x = "2023-08-31"
    start_x_plus_1 = "2023-08-30"
    end_x_plus_5 = "2023-09-03"

    historic_data = {
        'date_pred': pd.date_range(end=end_x, periods=5),
        'value': range(1, 6)
    }
    new_data = {
        'date': pd.date_range(start=start_x_plus_1, end=end_x_plus_5),
        'value': range(6, 11)
    }

    upai.dd.historic_predictions[pair] = pd.DataFrame(historic_data)

    new_pred_df = pd.DataFrame(new_data)
    dataframe = pd.DataFrame(new_data)

    # Action
    with patch('logging.Logger.warning') as mock_logger_warning:
        upai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    hist_pred_df = upai.dd.historic_predictions[pair]
    model_return_df = upai.dd.model_return_values[pair]

    assert hist_pred_df['date_pred'].iloc[-1] == pd.Timestamp(end_x_plus_5)
    assert 'date_pred' in hist_pred_df.columns
    assert hist_pred_df.shape[0] == 8

    # compare values in model_return_df with hist_pred_df
    assert (model_return_df["value"].values ==
            hist_pred_df.tail(len(dataframe))["value"].values).all()
    assert model_return_df.shape[0] == len(dataframe)

    # Ensure logger error is not called
    mock_logger_warning.assert_not_called()


def test_set_initial_return_values_warning(mocker, upai_conf):
    """
    Simple test of set_initial_return_values that hits the warning
    associated with leaving a UpAi bot offline so long that the
    exchange candles have no common date with the historic predictions
    """

    strategy = get_patched_upai_strategy(mocker, upai_conf)
    exchange = get_patched_exchange(mocker, upai_conf)
    strategy.dp = DataProvider(upai_conf, exchange)
    upai = strategy.upai
    upai.live = False
    upai.dk = UpaiDataKitchen(upai_conf)
    # Setup
    pair = "BTC/USD"
    end_x = "2023-08-31"
    start_x_plus_1 = "2023-09-01"
    end_x_plus_5 = "2023-09-05"

    historic_data = {
        'date_pred': pd.date_range(end=end_x, periods=5),
        'value': range(1, 6)
    }
    new_data = {
        'date': pd.date_range(start=start_x_plus_1, end=end_x_plus_5),
        'value': range(6, 11)
    }

    upai.dd.historic_predictions[pair] = pd.DataFrame(historic_data)

    new_pred_df = pd.DataFrame(new_data)
    dataframe = pd.DataFrame(new_data)

    # Action
    with patch('logging.Logger.warning') as mock_logger_warning:
        upai.dd.set_initial_return_values(pair, new_pred_df, dataframe)

    # Assertions
    hist_pred_df = upai.dd.historic_predictions[pair]
    model_return_df = upai.dd.model_return_values[pair]

    assert hist_pred_df['date_pred'].iloc[-1] == pd.Timestamp(end_x_plus_5)
    assert 'date_pred' in hist_pred_df.columns
    assert hist_pred_df.shape[0] == 10

    # compare values in model_return_df with hist_pred_df
    assert (model_return_df["value"].values == hist_pred_df.tail(
        len(dataframe))["value"].values).all()
    assert model_return_df.shape[0] == len(dataframe)

    # Ensure logger error is not called
    mock_logger_warning.assert_called()

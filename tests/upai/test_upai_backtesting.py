from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from uptrading.commands.optimize_commands import setup_optimize_configuration
from uptrading.enums import RunMode
from uptrading.exceptions import OperationalException
from uptrading.optimize.backtesting import Backtesting
from tests.conftest import (CURRENT_TEST_STRATEGY, get_args, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)


def test_upai_backtest_start_backtest_list(upai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('uptrading.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('uptrading.optimize.backtesting.history.load_data')
    mocker.patch('uptrading.optimize.backtesting.history.get_timerange', return_value=(now, now))

    patched_configuration_load_config_file(mocker, upai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '1m',
        '--strategy-list', CURRENT_TEST_STRATEGY
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re('Using --strategy-list with UpAi REQUIRES all strategies to have identical',
                      caplog)
    Backtesting.cleanup()


def test_upai_backtest_load_data(upai_conf, mocker, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('uptrading.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('uptrading.optimize.backtesting.history.load_data')
    mocker.patch('uptrading.optimize.backtesting.history.get_timerange', return_value=(now, now))
    backtesting = Backtesting(deepcopy(upai_conf))
    backtesting.load_bt_data()

    assert log_has_re('Increasing startup_candle_count for upai to.*', caplog)

    Backtesting.cleanup()


def test_upai_backtest_live_models_model_not_found(upai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch('uptrading.plugins.pairlistmanager.PairListManager.whitelist',
                 PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT']))
    mocker.patch('uptrading.optimize.backtesting.history.load_data')
    mocker.patch('uptrading.optimize.backtesting.history.get_timerange', return_value=(now, now))
    upai_conf["timerange"] = ""
    upai_conf.get("upai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, upai_conf)

    args = [
        'backtesting',
        '--config', 'config.json',
        '--datadir', str(testdatadir),
        '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'),
        '--timeframe', '5m',
        '--upai-backtest-live-models'
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(OperationalException,
                       match=r".* Historic predictions data is required to run backtest .*"):
        Backtesting(bt_config)

    Backtesting.cleanup()

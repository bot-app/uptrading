# flake8: noqa: F401
# isort: off
from uptrading.resolvers.iresolver import IResolver
from uptrading.resolvers.exchange_resolver import ExchangeResolver
# isort: on
# Don't import HyperoptResolver to avoid loading the whole Optimize tree
# from uptrading.resolvers.hyperopt_resolver import HyperOptResolver
from uptrading.resolvers.pairlist_resolver import PairListResolver
from uptrading.resolvers.protection_resolver import ProtectionResolver
from uptrading.resolvers.strategy_resolver import StrategyResolver

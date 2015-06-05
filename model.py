import pandas as pd
import numpy as np
import datetime
import blaze as bz
from cytoolz import sliding_window, count
from scipy.stats import chi2

import logging
from bokeh.models import ColumnDataSource

# def quantile(scenarios, level):
#     return np.percentile(scenarios, 100-level, interpolation='linear')

class VaR(object):
    """docstring for VaR"""
    def __init__(self, confidence_level):
        self.level = confidence_level

    def __call__(self, scenarios, neutral_scenario=0):
        pnls = scenarios - neutral_scenario
        return - np.percentile(pnls, 100-self.level, interpolation='linear'), pnls

    def likelihood_statistic(self, n_outliers, n_obs):
        p_obs = n_outliers * 1.0 / n_obs
        p_expected = 1. - self.level
        stat_expected = p_expected ** n_outliers * (1-p_expected) ** (n_obs-n_outliers)
        stat_obs = p_obs ** n_outliers * (1-p_obs) ** (n_obs - n_outliers)
        return -2 * np.log(stat_expected / stat_obs)

    def confidence(self, likelihood_stat):
        p_value = chi2.cdf(likelihood_stat, 1)
        return p_value

    @classmethod
    def traffic_light(cls, x, upper=0.99, lower=0.95):
        lights = np.ceil(np.clip((x - lower) / (upper-lower), 0., 1.01)).astype('int')
        return lights




class RiskFactor(object):
    """docstring for RiskFactor"""
    def __init__(self, ts):
        # super(RiskFactor, self).__init__()
        self.ts = ts

    def logreturns(self, n_days=1):
        self.ts['LogReturns'] = np.log( self.ts.Value.pct_change(periods=n_days) + 1)

    def devol(self, _lambda=0.06):
        _com = (1 - _lambda) / _lambda
        self.ts['Vola'] = pd.ewmstd( self.ts.LogReturns, com=_com, ignore_na=True)
        self.ts['DevolLogReturns'] = self.ts.LogReturns / self.ts.Vola

    def fhs(self, n_scenarios=250, start_date=None, end_date=None):
        x = sliding_window(n_scenarios+1, range(len(self.ts.index)))
        scenarios = np.zeros((len(self.ts.index), n_scenarios+1))
        for i, el in enumerate(x):
            l = list(el)
            cur_idx, hist_idx = l[-1], l[:-1]
            neutral = self.ts.Value.values[cur_idx]
            ret = self.ts.DevolLogReturns.values[hist_idx]
            vol = self.ts.Vola.values[cur_idx]
            scenarios[cur_idx, 1:] = self.scenario_values(ret, neutral, vol)
            scenarios[cur_idx, 0] = neutral
        return scenarios

    @classmethod
    def from_blaze(cls, filename, date_col='Date', value_col='Close'):
        df = bz.odo(filename, pd.DataFrame)[[date_col, value_col]] #[1000:1100]
        df = df.rename(columns = {value_col: 'Value'})
        ts = df.set_index(date_col)
        return cls(ts)

    @classmethod
    def scenario_values(cls, returns, neutral, current_vola):
        scenarios = neutral * np.exp(current_vola * returns)
        return scenarios



class CurrencyRiskFactor(RiskFactor):
    """docstring for CurrencyRiskFactor"""
    def __init__(self, *args):
        super(CurrencyRiskFactor, self).__init__(*args)

    @classmethod
    def from_blaze(clz, filename, date_col='Date', value_col='Rate'):
        return super(CurrencyRiskFactor, clz).from_blaze(filename, date_col=date_col, value_col=value_col)


class Future(object):
    """docstring for Future"""
    def __init__(self, ul, ccy):
        super(Future, self).__init__()
        self.ccy = ccy
        self.underlying = ul
        # TODO align risk factors

    def pv(self):
        pass

class StockModel(object):
    """docstring for StockModel"""
    def __init__(self):
        super(StockModel, self).__init__()
        file_name = "notebooks/db2.bcolz"
        self.df = bz.odo(file_name, pd.DataFrame)[['Date', 'Close']] #[1000:1100]
        self.devol()
        self.returns_df = None

    def devol(self, _lambda=0.06, n_days=1):
        _com = (1 - _lambda) / _lambda
        self.df['LogReturns'] = np.log(self.df.Close.pct_change(periods=n_days) + 1)
        self.df['Vola'] = pd.ewmstd( self.df.LogReturns, com=_com, ignore_na=True)[2:]
        self.df['DevolLogReturns'] = self.df.LogReturns / self.df.Vola
        self.df.set_index('Date', inplace=True)

    def compute_scenarios(self, d, n_scenarios=750):
        # identify returns
        dates = pd.to_datetime(d, unit='ms')
        max_date = dates[0].date()
        min_date = max_date.replace(year=max_date.year-3)

        logging.info('Computing returns between ') #, str(max_date), ' and ', str(min_date))
        self.returns_df = self.df[min_date:max_date].ix[-n_scenarios-1:]
        neutral, vola = self.returns_df.ix[max_date][['Close', 'Vola']]
        scenarios = neutral * np.exp( vola * self.returns_df.ix[:-1].DevolLogReturns )
        return scenarios, neutral

    def compute_var(self, scenarios, neutral_scenario, level=99.):
        pnls = scenarios - neutral_scenario
        return - np.percentile(pnls, 100-level, interpolation='linear'), pnls

    def compute_data_source(self):
        source = ColumnDataSource(self.df.reset_index()[2:])
        source.add(self.df[2:].LogReturns.ge(0).map(lambda x: "steelblue" if x else "red"), 'LogReturnsColor')
        source.add(self.df[2:].DevolLogReturns / 2., 'y_mids')
        return source

    def compute_histo_source(self, source, scenarios, bins=30):
        hist, edges = np.histogram(scenarios, density=True, bins=bins)
        source.data = dict(top=hist, bottom=0, left=edges[:-1], right = edges[1:])





#! /usr/bin/python3
from classification.features.feature_classifier import FeatureExtractor
import itertools
from scipy import optimize
from sklearn.metrics import mean_squared_error as mse


class HoltWintersCoeffsExtractor(FeatureExtractor):

    def extract_features(self, items):
        features=[self.extract_var_features(item) 
                  for item in items]
        return features

    def extract_var_features(self, item):
        axes_coeffs=[self.process_single_axis(item[axis])
                     for axis in ("x", "y", "z")]
        all_coeffs=itertools.chain.from_iterable(axes_coeffs)
        return list(all_coeffs)

    def process_single_axis(self, series):
        frame=series.to_frame()
        amount=len(series)
        values=series.values
        #values=series
        train, test=values[:amount*3//4], values[amount*3//4:]
        params=optimize.brute(call_hw, ((0, 1), (0, 1), (0, 1)), 
                              args=(series, ), Ns=10)
        print("found params: ", params)
        return params


def call_hw(params, *args):
    series=args[0]
    a, b, y=params
    season_len=10  # ?
    predictions=triple_exponential_smoothing(series, season_len, 
                                             a, b, y, 0)
    error=mse(series, predictions)
    return error


# author: https://grisha.org/blog/2016/02/17/triple-exponential-smoothing-forecasting-part-iii/

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result


def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

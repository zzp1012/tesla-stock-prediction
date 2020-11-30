# Dependency
# regular package
import datetime
import logging
import argparse
import copy

# data processing package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# statistics tools
from pandas_datareader import data as wb # datareader supports multiple financial database including yahoo and google
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

# set the logger
logging.basicConfig(
                    # filename = "logfile",
                    # filemode = "w+",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger("ARIMA")


def loss(y_pred, y_truth, loss_func):
    """
    y_pred: predicted values.
    y_truth: ground truth.
    loss_func: the function used to calculate loss.
    return loss value.
    """
    if loss_func == "mse":
        return mean_squared_error(y_truth, y_pred)
    elif loss_func == "r2":
        return r2_score(y_truth, y_pred)
    else:
        if loss_func != "rmse":
            logger.warning("The loss functin is illegal. Turn to default loss function: rmse")
        return np.sqrt(mean_squared_error(y_truth, y_pred))


def ARIMA_model(time_series, time_series_diff, diff_counts, args):
    """
    times_seris: original time_series, pd.Dataframe.
    time_series_diff: stationary time_series after diff. 
    diff_counts: counts of diff when the time_series become stationary.
    args: arguments parsed before.
    return fitted ARIMA model.
    """
    # check if args.ic is illegal.
    if args.ic not in ["bic", "aic"]:
        logger.warning("The information criteria is illegal. Turn to default ic: BIC")
        args.ic = "bic"

    evaluate = sm.tsa.arma_order_select_ic(time_series_diff,
                                           ic = args.ic,
                                           trend = "c",
                                           max_ar = args.max_ar,
                                           max_ma = args.max_ma)
    # get the parameter for ARIMA model.
    min_order = evaluate[args.ic + "_min_order"]
    # construct the ARIMA model.
    logger.info("ARIMA parameters: " + str(tuple(min_order[0], diff_counts, min_order[1])))
    model = ARIMA(timeseries, order=(min_order[0], diff_counts, min_order[1])) # d is the order of diff, which we have done that perviously.

    # check the value of convergence tol.
    if args.tol > 0.01:
        logger.warning("The convergence tolerance is too large. Turn to use default value: 1e-8")
        args.tol = 1e-8

    if args.method not in ["css-mle", "mle", "css"]:
        logger.warning("The likelihood function is illegal. Turn to default choice: css-mle")
        args.method = "css-mle"
    
    model_fit = model.fit(disp = False, 
                          method = args.method,
                          trend = "c", # Some posts' experimentation suggests that ARIMA models may be less likely to converge with the trend term disabled, especially when using more than zero MA terms.
                          transparams = True,
                          solver = "lbfgs", # we turn to use this one, which gives the best RMSE & executation time.
                          tol = args.tol, # The convergence tolerance. Default is 1e-08.
                          )
    return model_fit


def diff(time_series):
    """
    times_seris: time_series, pd.Dataframe.
    return stationary time_series, counts of diff when the time_series become stationary.
    """
    counts = 0 # indicating how many times the series diffs.
    copy_series = copy.deepcopy(time_series)
    # keep diff until ADF test's p-value is smaller than 1%.
    while ADF(copy_series.tolist()[1]) > 0.01:
        copy_series = copy_series.diff(1)
        copy_series = copy_series.fillna(0)
        counts += 1
    return copy_series, counts


def decompose(time_series, season_period):
    """
    times_seris: time_series, pd.Dataframe.
    season_period: period of seasonality, float.
    return the decomposition of the time_series including trend, seasonal, residual.
    """    
    decomposition = seasonal_decompose(time_series, 
                                       model='additive', 
                                       extrapolate_trend='freq', 
                                       period=season_period) # additive model is the default choice.
    return decomposition.trend, decomposition.seasonal, decomposition.resid


def data_loader(tickers, month):
    """
    tickers: list of tuples, containing info of ticker and data sources.
    return the dataframes according to tickers from corresponding sources.
    """
    # get the start date and end date
    start_date = datetime.date.today() + relativedelta(months = -month)
    end_date = datetime.date.today()
    # fetching data frames
    stock_dfs = list()
    for ticker, source in tickers:
        if source == "fred":
            df = pd.DataFrame(wb.DataReader(ticker, 
                                            data_source = source, 
                                            start = start_date + relativedelta(days = -1), 
                                            end = end_date))
        else:
            df = pd.DataFrame(wb.DataReader(ticker, 
                                            data_source = source, 
                                            start = start_date, 
                                            end = end_date))
        stock_dfs.append(df)
    return stock_dfs


def add_args():
    """
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser(description="TSLA-ARIMA")

    # hyperparameters setting
    parser.add_argument('--month', type=int, default=3,
                        help='The data ranges from several months ago to now. Default 4. Suggest that not too long period to avoid outliers in data.')

    parser.add_argument('--period', type=int, default=4,
                        help='The seasonal period. Default 4.')

    parser.add_argument('--split_ratio', type=float, default=0.7,
                        help='Data splitting ratio. Default 0.7.')

    parser.add_argument('--max_ar', type=int, default=4,
                        help='Maximum number of AR lags to use. Default 4.')

    parser.add_argument('--max_ma', type=int, default=2,
                        help='Maximum number of MA lags to use. Default 2.')

    parser.add_argument('--ic', type=str, default="BIC",
                        help='Information criteria to report. Either a single string or a list of different criteria is possible. Default BIC.')

    parser.add_argument('--tol', type=float, default=1e-8,
                        help='The convergence tolerance. Default is 1e-08.')

    parser.add_argument('--method', type=str, default="css-mle",
                        help='This is the loglikelihood to maximize. Default is css-mle.')

    parser.add_argument('--loss', type=str, default="rmse",
                        help='The loss function. Default rmse.')

    # set if using debug mod
    parser.add_argument("-v", "--verbose", action= "store_true", dest= "verbose", 
                        help= "Enable debug info output. Default false.")

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = add_args()

    # set the level of logger
    logger.setLevel(logging.DEBUG)
    if not args.verbose:
        logger.setLevel(logging.INFO)
    logger.debug("--------DEBUG enviroment start---------")
    
    # show the hyperparameters
    logger.info("---------hyperparameter setting---------")
    logger.info(args)

    # data fetching
    logger.info("----------Data start fetching-----------")
    tickers = \
    [
        ("TSLA", "yahoo"), # 0, TESLA Stock
    ]
    tsla_df = data_loader(tickers, args.month)[0] # get dataframes from "yahoo" finance.
    tsla_close = tsla_df["Close"]
    logger.info("----------Data stop fetching------------")

    # data cleaning
    if np.sum(tsla_close.isnull()) > 0:
        logger.debug("The time series contain missing values & we use interpolation to resolve this issue")
        tsla_close = tsla_close.interpolate(method='polynomial', order=2, limit_direction='forward', axis=0)
    # Then, if there is still some missing values, we simply drop this value.abs
    tsla_close = tsla_close.dropna()

    # data splitting
    train = tsla_close[0:round(len(tsla_close) * args.split_ratio)]
    test = tsla_close[round(len(tsla_close) * args.split_ratio):len(tsla_close)]

    # time serise decomposition
    trend, seasonal, resiudal = decompose(train, args.period)

    # difference
    logger.debug("-----------------Diff-----------------")
    trend_diff, trend_diff_counts = diff(trend)
    logger.debug("trend diff counts: " + str(trend_diff_counts))
    residual_diff, residual_diff_counts = diff(residual)
    logger.debug("residual diff counts: " + str(residual_diff_counts))
    
    # ARIMA model
    logger.info("-----------ARIMA construction----------")
    trend_model_fit = ARIMA_model(trend, trend_diff, trend_diff_counts, args)
    residual_model_fit = ARIMA_model(residual, residual_diff, residual_diff_counts, args)

    # model summary
    logger.debug("---------trend model summary----------")
    logger.debug(trend_model_fit.summary())
    logger.debug("---------resid model summary----------")
    logger.deubg(residual_model_fit.summary())

    # loss calculation
    logger.info("-----------loss calculation------------")
    trend_fit_seq = trend_model_fit.fittedvalues
    residual_fit_seq = residual_model_fit.fittedvalues
    fit_seq = seasonal + trend_fit_seq + residual_fit_seq

    training_loss = loss(np.array(fit_seq), np.array(train), args.loss)
    logger.info("Training loss: " + str(training_loss))

if __name__ == "__main__":
    main()
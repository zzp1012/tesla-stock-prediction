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

# ignore warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", HessianInversionWarning)

# set the logger
logging.basicConfig(
                    # filename = "logfile",
                    # filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.ERROR)
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


def model_predict(trend_model_fit, residual_model_fit, 
                  trend, residual, seasonal, 
                  trend_diff_counts, residual_diff_counts, 
                  if_pred, start, end, period):
    """
    trend_model_fit: ARIMA model after fit the trend.
    residual_model_fit: ARIMA model after fit the residual.
    trend: time series of trend.
    residual: time series of residual.
    seasonal: time series of seasonal.
    trend_diff_counts: int value indicating counts of diff for trend.
    residual_diff_counts: int values indicating counts of diff for residual.
    if_pred: boolen value indicating whether to predict or not. True presents predict, False means fit.
    start: string value indicating start date.
    end: string value indicating end date.
    period: int value indicating the period of seasonal.
    return predicted sequence.
    """
    if if_pred:
        date_after_train = str(trend.index.tolist()[-1] + relativedelta(days=1))
        trend_pred_seq = np.array(trend_model_fit.predict(start = date_after_train, 
                                                          end = end,
                                                          dynamic = True)) # The dynamic keyword affects in-sample prediction. 
        trend_pred_seq = np.array(np.concatenate((np.array(trend.diff(trend_diff_counts).fillna(0)),
                                                  trend_pred_seq)))
        residual_pred_seq = np.array(residual_model_fit.predict(start = date_after_train,
                                                                end = end,
                                                                dynamic = True))
        residual_pred_seq = np.array(np.concatenate((np.array(residual.diff(residual_diff_counts).fillna(0)),
                                                     residual_pred_seq)))
        # find the the corresponding seasonal sequence.
        pred_period = (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days + 1
        seasonal_pred_seq = list(seasonal[len(seasonal) - period:len(seasonal)]) * (round((pred_period) / period) + 1) 
        seasonal_pred_seq = np.array(seasonal_pred_seq[0:pred_period])
    else:
        trend_pred_seq = np.array(trend_model_fit.fittedvalues)
        residual_pred_seq = np.array(residual_model_fit.fittedvalues)
        seasonal_pred_seq = np.array(seasonal)

    if trend_diff_counts > 0:
        trend_pred_seq = trend_pred_seq + trend[0]
    if residual_diff_counts > 0:
        residual_pred_seq = residual_pred_seq + residual[0]
    
    while trend_diff_counts > 0 or residual_diff_counts > 0:
        if trend_diff_counts > 0:
            trend_pred_seq.cumsum()
            trend_diff_counts -= 1
        if residual_diff_counts > 0:
            residual_pred_seq.cumsum()
            residual_diff_counts -= 1
    if if_pred:
        return trend_pred_seq[len(trend_pred_seq)-len(seasonal_pred_seq):len(trend_pred_seq)] + \
               residual_pred_seq[len(residual_pred_seq)-len(seasonal_pred_seq):len(residual_pred_seq)] + \
               seasonal_pred_seq
    else:
        return trend_pred_seq + residual_pred_seq + seasonal_pred_seq


def ARIMA_model(time_series_diff, args):
    """
    time_series_diff: stationary time_series after diff. 
    args: arguments parsed before.
    return fitted ARIMA model, parameters for ARIMA model.
    """
    # check if args.ic is illegal.
    if args.ic not in ["bic", "aic"]:
        logger.warning("The information criteria is illegal. Turn to default ic: BIC")
        args.ic = "bic"

    # check the value of convergence tol.
    if args.tol > 0.01:
        logger.warning("The convergence tolerance is too large. Turn to use default value: 1e-8")
        args.tol = 1e-8
    
    # check the likelihood function used.
    if args.method not in ["css-mle", "mle", "css"]:
        logger.warning("The likelihood function is illegal. Turn to default choice: css-mle")
        args.method = "css-mle"

    evaluate = sm.tsa.arma_order_select_ic(time_series_diff,
                                           ic = args.ic,
                                           trend = "c",
                                           max_ar = args.max_ar,
                                           max_ma = args.max_ma)
    # get the parameter for ARIMA model.
    min_order = evaluate[args.ic + "_min_order"]

    # initial the success_flag to false
    success_flag = False
    while not success_flag:
        # construct the ARIMA model.
        model = ARIMA(time_series_diff, order=(min_order[0], 0, min_order[1])) # d is the order of diff, which we have done that perviously.
        # keep finding initial parameters until convergence.
        try:
            model_fit = model.fit(disp = False, 
                                start_params = np.random.rand(min_order[0] + min_order[1] + 1),
                                method = args.method,
                                trend = "c", # Some posts' experimentation suggests that ARIMA models may be less likely to converge with the trend term disabled, especially when using more than zero MA terms.
                                transparams = True,
                                solver = "lbfgs", # we turn to use this one, which gives the best RMSE & executation time.
                                tol = args.tol, # The convergence tolerance. Default is 1e-08.
                                )
            success_flag = True
        except ValueError:
            logger.warning("ValueError occurs, choosing another starting parameters.")
        except np.linalg.LinAlgError:
            logger.warning("np.linalg.LinAlgError occurs, choosing another starting parameters.")

    return model_fit, min_order


def diff(time_series):
    """
    times_seris: time_series, pd.Dataframe.
    return stationary time_series, counts of diff when the time_series become stationary.
    """
    counts = 0 # indicating how many times the series diffs.
    copy_series = copy.deepcopy(time_series)
    # keep diff until ADF test's p-value is smaller than 1%.
    while ADF(copy_series.tolist())[1] > 0.01:
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
                                       model='additive', # additive model is the default choice.
                                       extrapolate_trend='freq', 
                                       period=season_period) 
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
        elif source == "yahoo":
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

    parser.add_argument('--period', type=int, default=5,
                        help='The seasonal period. Default 5.')

    parser.add_argument('--split_ratio', type=float, default=0.7,
                        help='Data splitting ratio. Default 0.7.')

    parser.add_argument('--max_ar', type=int, default=4,
                        help='Maximum number of AR lags to use. Default 4.')

    parser.add_argument('--max_ma', type=int, default=2,
                        help='Maximum number of MA lags to use. Default 2.')

    parser.add_argument('--ic', type=str, default="bic",
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

    # set if plots
    parser.add_argument("-p", "--plot", action= "store_true", dest= "plot", 
                        help= "Enable plots. Default false.")

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
    logger.info("-------------Data fetching-------------")
    tickers = \
    [
        ("TSLA", "yahoo"), # 0, TESLA Stock
    ]
    # check if data range is legal.
    if  args.month <= 0 or args.month > 24:
        logger.warning("The data range is illegal. Turn to use default 3")
        args.month = 3
    tsla_df = data_loader(tickers, args.month)[0] # get dataframes from "yahoo" finance.
    tsla_close = tsla_df["Close"].resample('D').ffill() # fullfill the time series.
    logger.debug(tsla_close)

    # data cleaning
    logger.info("-------------Data cleaning-------------")
    if np.sum(tsla_close.isnull()) > 0:
        logger.debug("The time series contain missing values & we use interpolation to resolve this issue")
        tsla_close = tsla_close.interpolate(method='polynomial', order=2, limit_direction='forward', axis=0)
    # Then, if there is still some missing values, we simply drop this value.abs
    tsla_close = tsla_close.dropna()

    # data splitting
    logger.info("-----------Data splitting--------------")
    # check if split_ratio legal.
    if args.split_ratio > 1 or round(len(tsla_close) * args.split_ratio) <= 0:
        logger.warning("Splitting ratio is illegal. Turn to use default 0.7")
        args.split_ratio = 0.7
    train = tsla_close[0:round(len(tsla_close) * args.split_ratio)]
    test = tsla_close[round(len(tsla_close) * args.split_ratio):len(tsla_close)]

    # time serise decomposition
    logger.info("------------decomposition--------------")
    # check if period is legal.
    if args.period < 2:
        logger.warning("Seasonal period is illegal. Turn to use default 5.")
        args.period = 5
    trend, seasonal, residual = decompose(train, args.period)

    # difference
    logger.debug("-----------------Diff-----------------")
    trend_diff, trend_diff_counts = diff(trend)
    logger.debug("trend diff counts: " + str(trend_diff_counts))
    residual_diff, residual_diff_counts = diff(residual)
    logger.debug("residual diff counts: " + str(residual_diff_counts))
    
    # ARIMA model
    logger.info("-----------ARIMA construction----------")
    trend_model_fit, trend_model_order = ARIMA_model(trend_diff, args)
    logger.info("Trend model parameters: " + str(tuple([trend_model_order[0],
                                                        trend_diff_counts,
                                                        trend_model_order[1]])))
    residual_model_fit, residual_model_order = ARIMA_model(residual_diff, args)
    logger.info("Residual model parameters: " + str(tuple([residual_model_order[0],
                                                          residual_diff_counts,
                                                          residual_model_order[1]])))

    # model summary
    logger.debug("---------trend model summary----------")
    logger.debug(trend_model_fit.summary())
    logger.debug("---------resid model summary----------")
    logger.debug(residual_model_fit.summary())

    # loss calculation
    logger.info("-----------Loss calculation------------")
    fit_seq = model_predict(trend_model_fit, residual_model_fit,
                            trend, residual, seasonal, 
                            trend_diff_counts, residual_diff_counts, 
                            False, "", "", args.period)
    logger.debug(fit_seq)

    # calculate training loss
    training_loss = loss(fit_seq, np.array(train), args.loss)
    logger.info("Training loss: " + str(training_loss))

    if list(test):
        test_start = str(test.index.tolist()[0]).replace(" 00:00:00", "")
        test_end = str(test.index.tolist()[-1]).replace(" 00:00:00", "")
        pred_seq = model_predict(trend_model_fit, residual_model_fit,
                                trend, residual, seasonal, 
                                trend_diff_counts, residual_diff_counts, 
                                True, test_start, test_end, args.period)
        logger.debug(pred_seq) 

        testing_loss = loss(pred_seq, np.array(test), args.loss)
        logger.info("Testing loss: " + str(testing_loss))

    # prediction
    logger.info("--------------prediction---------------")
    prediction = model_predict(trend_model_fit, residual_model_fit,
                               trend, residual, seasonal, 
                               trend_diff_counts, residual_diff_counts, 
                               True, "2020-12-07", "2020-12-11", args.period)
    logger.info("2020-12-07 predicted value: " + str(prediction[0]))
    logger.info("2020-12-08 predicted value: " + str(prediction[1]))
    logger.info("2020-12-09 predicted value: " + str(prediction[2]))
    logger.info("2020-12-10 predicted value: " + str(prediction[3]))
    logger.info("2020-12-11 predicted value: " + str(prediction[4]))
    logger.info("--------------Process ends-------------")

if __name__ == "__main__":
    main()
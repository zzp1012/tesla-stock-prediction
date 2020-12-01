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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # plot acf and pacf.

# ignore warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", HessianInversionWarning)
warnings.simplefilter("ignore", RuntimeWarning)

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
        # get the first date after the last date in train.
        date_after_train = str(trend.index.tolist()[-1] + relativedelta(days=1))
        # get the trend predicted sequence from the start of start to end
        trend_pred_seq = np.array(trend_model_fit.predict(start = date_after_train, 
                                                          end = end,
                                                          dynamic = True)) # The dynamic keyword affects in-sample prediction. 
        trend_pred_seq = np.array(np.concatenate((np.array(trend.diff(trend_diff_counts).fillna(0)),
                                                  trend_pred_seq)))
        # get the residual predicted sequence from the start of start to end
        residual_pred_seq = np.array(residual_model_fit.predict(start = date_after_train,
                                                                end = end,
                                                                dynamic = True))
        residual_pred_seq = np.array(np.concatenate((np.array(residual.diff(residual_diff_counts).fillna(0)),
                                                     residual_pred_seq)))
        # find the the corresponding seasonal sequence.
        pred_period = (datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(date_after_train, '%Y-%m-%d %H:%M:%S')).days + 1
        seasonal_pred_seq = list(seasonal[len(seasonal) - period:]) * (round((pred_period) / period) + 1) 
        seasonal_pred_seq = np.array(seasonal_pred_seq[0:pred_period])
    else:
        trend_pred_seq = np.array(trend_model_fit.fittedvalues)
        residual_pred_seq = np.array(residual_model_fit.fittedvalues)
        seasonal_pred_seq = np.array(seasonal)
    
    while trend_diff_counts > 0 or residual_diff_counts > 0:
        if trend_diff_counts > 0:
            trend_pred_seq.cumsum()
            trend_diff_counts -= 1
            if trend_diff_counts == 0:
                trend_pred_seq = trend_pred_seq + trend[0]
        if residual_diff_counts > 0:
            residual_pred_seq.cumsum()
            residual_diff_counts -= 1
            if residual_diff_counts == 0:
                residual_pred_seq = residual_pred_seq + residual[0]

    if if_pred:
        pred_period = (datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')).days + 1
        return trend_pred_seq[len(trend_pred_seq)-pred_period:] + \
               residual_pred_seq[len(residual_pred_seq)-pred_period:] + \
               seasonal_pred_seq[len(seasonal_pred_seq)- pred_period:]
    else:
        return trend_pred_seq + residual_pred_seq + seasonal_pred_seq


def ARIMA_model(time_series_diff, args, name):
    """
    time_series_diff: stationary time_series after diff. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return fitted ARIMA model, parameters for ARIMA model.
    """
    if args.plot and name in ["trend_diff", "residual_diff"]:
        fig, axes = plt.subplots(1,2, figsize=(16,3), dpi= 100)
        plot_acf(time_series_diff.tolist(), lags=min(50, len(time_series_diff) - 1), ax=axes[0])
        plot_pacf(time_series_diff.tolist(), lags=min(50, len(time_series_diff) - 1), ax=axes[1])
        plt.savefig(name + "_acf_pacf.png")
        plt.close()

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
        except:
            logger.warning("Error occurs, try another starting parameters.")
            pass

    return model_fit, min_order


def diff(time_series, if_plot, name, if_diff):
    """
    times_seris: time_series, pd.Dataframe.
    if_plot: boolen value indicating whether to plot.
    name: string value indicating name of the time series.
    if_diff: boolen value indicating whether to diff.
    return stationary time_series, counts of diff when the time_series become stationary.
    """
    counts = 0 # indicating how many times the series diffs.
    copy_series = copy.deepcopy(time_series)
    
    # directly return if_diff False.
    if not if_diff:
        return copy_series, counts
    
    # keep diff until ADF test's p-value is smaller than 1%.
    while ADF(copy_series.tolist())[1] > 0.01:
        logger.info("time " + str(counts) + " ADF test: " + str(ADF(copy_series.tolist())))
        copy_series = copy_series.diff(1)
        copy_series = copy_series.fillna(0)
        counts += 1
    
    logger.info("time " + str(counts) + " ADF test: " + str(ADF(copy_series.tolist())))

    # plot diff and original time series in one graph.
    if if_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(time_series, label='Original', color='blue')
        plt.plot(copy_series, label='Diff' + str(counts), color='red')
        plt.legend(loc='best')
        plt.savefig(name + "_diff.png")
        plt.close()
    
    return copy_series, counts


def decompose(time_series, season_period, if_plot):
    """
    times_seris: time_series, pd.Dataframe.
    season_period: period of seasonality, float.
    if_plot: boolen value indicating whether to plot.
    return the decomposition of the time_series including trend, seasonal, residual.
    """    
    # find the filt for seasonal_decompose
    if season_period % 2 == 0:  # split weights at ends
        filt = np.array([.5] + [1] * (season_period - 1) + [.5]) / season_period
    else:
        filt = np.repeat(1. / season_period, season_period)
    logger.info("seasonal_decompose filter: " + str(filt))

    decomposition = seasonal_decompose(time_series, 
                                       model = 'additive', # additive model is the default choice. 
                                                           # We tried "multiplicative" but it is totally meaningless.
                                       two_sided = True,
                                       filt = filt,
                                       extrapolate_trend = 'freq',
                                       period = season_period) 
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # plot the time series decomposition
    if if_plot:
        plt.figure(figsize=(12, 7))
        plt.subplot(411)
        plt.title("seasonal decomposition")
        plt.plot(time_series, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonarity')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residual')
        plt.legend(loc='best')
        plt.savefig("decomposition.png")
        plt.close()

    return trend, seasonal, residual


def app_entropy(U, m = 2, r = 3) -> float:
    """
    U: the time series.
    m: an integer representing the length of compared run of data. Default 2.
    r: a positive real number specifying a filtering level. Default 3.
    return approximate entroy.
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))


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

    parser.add_argument('--period', type=int, default=7,
                        help='The seasonal period. Default 7.')

    parser.add_argument('--split_ratio', type=float, default=0.7,
                        help='Data splitting ratio. Default 0.7.')

    parser.add_argument('--max_ar', type=int, default=4,
                        help='Maximum number of AR lags to use. Default 4.')

    parser.add_argument('--max_ma', type=int, default=4,
                        help='Maximum number of MA lags to use. Default 4.')

    parser.add_argument('--ic', type=str, default="bic",
                        help='Information criteria to report. Either a single string or a list of different criteria is possible. Default BIC.')

    parser.add_argument('--tol', type=float, default=1e-8,
                        help='The convergence tolerance. Default is 1e-08.')

    parser.add_argument('--method', type=str, default="css-mle",
                        help='This is the loglikelihood to maximize. Default is css-mle.')

    parser.add_argument('--loss', type=str, default="rmse",
                        help='The loss function. Default rmse.')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default 0.')
    
    parser.add_argument("-l", "--log", action="store_true", dest="log", 
                        help= "Enable log transformation. Default false.")

    parser.add_argument("-d", "--diff", action="store_false", dest="diff", 
                        help= "Disable difference. Default True.")                    

    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
                        help= "Enable debug info output. Default false.")

    # set if plots
    parser.add_argument("-p", "--plot", action="store_true", dest="plot", 
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

    # set the random seed
    np.random.seed(args.seed)

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

    # data cleaning
    logger.info("-------------Data cleaning-------------")
    if np.sum(tsla_close.isnull()) > 0:
        logger.debug("The time series contain missing values & we use interpolation to resolve this issue")
        tsla_close = tsla_close.interpolate(method='polynomial', order=2, limit_direction='forward', axis=0)
    # Then, if there is still some missing values, we simply drop this value.abs
    tsla_close = tsla_close.dropna()
    logger.debug(tsla_close)

    # plot the graph describe tsla close
    if args.plot:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.plot(tsla_close, label = "Series")
        plt.plot(tsla_close.rolling(int(.05 * len(tsla_close))).mean(), '--', 
                 label = "Rolling mean")
        plt.plot(tsla_close.rolling(int(.05 * len(tsla_close))).std(), ":",
                 label = "Rolling Std")
        plt.legend(loc = "best")
        plt.savefig("tesla_description.png")

    # if log transformation
    if args.log:
        tsla_close = tsla_close.apply(np.log) # log transformation

    # estimate the forecastability of a time series:
    #   Approximate entropy is a technique used to quantify the amount of regularity and the unpredictability of fluctuations over time-series data. 
    #   Smaller values indicates that the data is more regular and predictable.
    logger.info("The approximate entropy: " + str(app_entropy(U = np.array(tsla_close), r = 0.2*np.std(np.array(tsla_close)))))

    # data splitting
    logger.info("-------------Data splitting------------")
    # check if split_ratio legal.
    if args.split_ratio > 1 or round(len(tsla_close) * args.split_ratio) <= 0:
        logger.warning("Splitting ratio is illegal. Turn to use default 0.7")
        args.split_ratio = 0.7
    train = tsla_close[0:round(len(tsla_close) * args.split_ratio)]
    test = tsla_close[round(len(tsla_close) * args.split_ratio):]

    # time serise decomposition
    logger.info("-------------decomposition-------------")
    # check if period is legal.
    if args.period < 2:
        logger.warning("Seasonal period is illegal. Turn to use default 7.")
        args.period = 7
    trend, seasonal, residual = decompose(train, args.period, args.plot)

    # difference
    logger.debug("-----------------Diff-----------------")
    trend_diff, trend_diff_counts = diff(trend, args.plot, "trend", args.diff)
    logger.debug("trend diff counts: " + str(trend_diff_counts))
    residual_diff, residual_diff_counts = diff(residual, args.plot, "residual", args.diff)
    logger.debug("residual diff counts: " + str(residual_diff_counts))
    
    # ARIMA model
    logger.info("-----------ARIMA construction----------")
    trend_model_fit, trend_model_order = ARIMA_model(trend_diff, args, "trend_diff")
    logger.info("Trend model parameters: " + str(tuple([trend_model_order[0],
                                                        trend_diff_counts,
                                                        trend_model_order[1]])))
    residual_model_fit, residual_model_order = ARIMA_model(residual_diff, args, "residual_diff")
    logger.info("Residual model parameters: " + str(tuple([residual_model_order[0],
                                                          residual_diff_counts,
                                                          residual_model_order[1]])))

    # model summary
    try:
        logger.debug("---------trend model summary----------")
        logger.debug(trend_model_fit.summary())
    except:
        logger.warning("Error occurs in summary, simply skip")
        pass    
    try:
        logger.debug("---------resid model summary----------")
        logger.debug(residual_model_fit.summary())
    except:
        logger.warning("Error occurs in summary, simply skip")
        pass

    if args.plot:
        # residual plots of trend model
        trend_model_fit.resid.plot()
        plt.savefig("resid_plt_trend.png")
        plt.close()
        trend_model_fit.resid.plot(kind='kde')
        plt.savefig("kde_resid_plt_trend.png")
        plt.close()
        
        # residual plots of residual model
        residual_model_fit.resid.plot()
        plt.savefig("resid_plt_residual.png")
        plt.close()
        residual_model_fit.resid.plot(kind='kde')
        plt.savefig("kde_resid_plt_residual.png")
        plt.close()

    logger.debug("-----trend model residual describe----")
    logger.debug(trend_model_fit.resid.describe()) # describe the dataframe 
    logger.debug("-----resid model residual describe----")
    logger.debug(residual_model_fit.resid.describe()) # describe the dataframe

    # loss calculation
    logger.info("-----------Loss calculation------------")
    fit_seq = model_predict(trend_model_fit, residual_model_fit,
                            trend, residual, seasonal, 
                            trend_diff_counts, residual_diff_counts, 
                            False, "", "", args.period)
    if args.log:
        fit_seq = np.exp(fit_seq)
        train = train.apply(np.exp)
    logger.debug(fit_seq)

    # calculate training loss
    training_loss = loss(fit_seq, np.array(train), args.loss)
    logger.info("Training loss: " + str(training_loss))

    # plot train and fitted values in one graph.
    if args.plot:
        plt.figure()
        plt.plot(fit_seq, color = 'red', label = 'fit')
        plt.plot(np.array(train), color = 'blue', label = 'train')
        plt.legend(loc = 'best')
        plt.savefig('fit_vs_train.png')
        plt.close()

    if list(test):
        pred_seq = model_predict(trend_model_fit, residual_model_fit,
                                trend, residual, seasonal, 
                                trend_diff_counts, residual_diff_counts, 
                                True, str(test.index.tolist()[0]), str(test.index.tolist()[-1]), args.period)
        if args.log:
            pred_seq = np.exp(pred_seq)
            test = test.apply(np.exp)
        logger.debug(pred_seq) 

        # calculate testing loss
        testing_loss = loss(pred_seq, np.array(test), args.loss)
        logger.info("Testing loss: " + str(testing_loss))

        # plot test and predicted value in one graph.
        if args.plot:
            plt.figure()
            plt.plot(pred_seq, color = "red", label = "pred")
            plt.plot(np.array(test), color = "blue", label = "test")
            plt.legend(loc="best")
            plt.savefig("pred_vs_test.png")
            plt.close()

    # plot several models performance comparison on train set.
    if args.plot:
        logger.info("-----------Model Comparison------------")
        plt.figure()
        # actual value 
        plt.plot(np.array(train), color = 'blue',
                 label = "actual")
        # auto-ARIMA with seasonal decompostion
        plt.plot(fit_seq[1:], color = 'green',
                 label = 'ARIMA with seasonal decomposition')
        # simple auto-ARIMA
        auto_arima_model_fit, _ = ARIMA_model(train, args, "auto_arima")
        plt.plot(np.array(auto_arima_model_fit.fittedvalues), color = 'yellow' , 
                label = 'Auto ARIMA')
        # auto-ARIMA with log transfromation.
        auto_log_arima_fit, _ = ARIMA_model(train.apply(np.log), args, "auto_arima")
        plt.plot(np.array(auto_log_arima_fit.fittedvalues.apply(np.exp)), color = 'brown' , label = 'Auto ARIMA with log')
        # rolling mean
        plt.plot(np.array(train.rolling(int(.05 * len(train))).mean()), '--', 
                 label = "Rolling mean")
        # ordinary arima
        plt.plot(np.array(ARIMA(train, (1, 0, 1)).fit(disp = 0).fittedvalues), color = "coral",
                 label = "Ordinary ARIMA")
        plt.legend(loc = "best")
        plt.xlabel("days from " + str(train.index.tolist()[0]).replace(" 00:00:00", ""))
        plt.ylabel("stock prices")
        plt.title("Actual Stock Price Compared with Forecasted Stock Price")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.close()

        if list(test):
            # calculate testing loss
            loss_dict = dict()
            # auto-ARIMA with seasonal decompostion
            loss_dict["auto sarima"] = testing_loss
            # simple auto-ARIMA
            loss_dict["auto arima"] = loss(np.array(auto_arima_model_fit.predict(start = str(test.index.tolist()[0]),
                                                                                 end = str(test.index.tolist()[-1]),
                                                                                 dynamic = True)),
                                           np.array(test), args.loss)
            # auto-ARIMA with log transfromation.
            loss_dict["auto arima log"] = loss(np.array(auto_log_arima_fit.predict(start = str(test.index.tolist()[0]),
                                                                                   end = str(test.index.tolist()[-1]),
                                                                                   dynamic = True).apply(np.exp)),
                                               np.array(test), args.loss)
            # ordinary arima
            loss_dict["arima"] = loss(np.array(ARIMA(train, (1, 0, 1)).fit(disp = 0).predict(start = str(test.index.tolist()[0]),
                                                                                                 end = str(test.index.tolist()[-1]),
                                                                                                 dynamic = True)),
                                               np.array(test), args.loss)
            logger.info(loss_dict)
            plt.figure(figsize=(12,6))
            loss_df = pd.DataFrame.from_dict(loss_dict, orient='index')
            plt.bar(loss_df.index.tolist(), loss_df.iloc[:, 0])
            plt.ylabel("RMSE")
            plt.legend('')
            plt.title("RMSE for Difference Models on Test Data")
            plt.savefig("RMSE_model_comparison.png")
            plt.close()

    # prediction
    logger.info("--------------prediction---------------")
    prediction = model_predict(trend_model_fit, residual_model_fit,
                               trend, residual, seasonal, 
                               trend_diff_counts, residual_diff_counts, 
                               True, "2020-12-07 00:00:00", "2020-12-11 00:00:00", args.period)
    if args.log:
        prediction = np.exp(prediction)
    logger.info("2020-12-07 predicted value: " + str(prediction[0]))
    logger.info("2020-12-08 predicted value: " + str(prediction[1]))
    logger.info("2020-12-09 predicted value: " + str(prediction[2]))
    logger.info("2020-12-10 predicted value: " + str(prediction[3]))
    logger.info("2020-12-11 predicted value: " + str(prediction[4]))
    logger.info("--------------Process ends-------------")

if __name__ == "__main__":
    main()
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
from arch import arch_model # import arch_model.
# check white noise. If p-value smaller than 0.05, then we are confident that the time series are not while noise.
from statsmodels.stats.diagnostic import acorr_ljungbox
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
warnings.simplefilter("ignore", FutureWarning)

# set the logger
logging.basicConfig(
                    # filename = "logfile",
                    # filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.ERROR)
logger = logging.getLogger("GARCH")


def plot_pvalue(pvalue, name):
    """
    pvalue:
    name: used for later label and title.
    effect:
    """
    plt.figure()
    plt.plot(pvalue)
    plt.plot(list(range(len(pvalue))), [0.05] * len(pvalue))
    plt.title(name.replace("_", " ") + " Test, P Value Plot")
    plt.savefig(name + "_test_pvalue.png")
    plt.close()


def plot_residual(resid, name):
    """
    resid:
    name: used for later label and title.
    effect:
    """
    resid.plot()
    plt.title(name + " Residual Plot")
    plt.savefig(name + "_resid_plt.png")
    plt.close()
    resid.plot(kind='kde')
    plt.title(name + " KDE Residual Plot")
    plt.savefig(name + "_kde_resid_plt.png")
    plt.close()

def plot_pred_vs_truth(y_pred, y_truth, label1, label2):
    """
    """
    plt.figure()
    plt.plot(y_pred, color='red', label=label1)
    plt.plot(y_truth, color ='blue', label=label2)
    plt.legend(loc='best')
    plt.title(label1 + " v.s. " + label2)
    plt.savefig(label1 + "_vs_" + label2 + ".png")
    plt.close()


def plot_description(time_series, name):
    """
    time_series: 
    name: used for later label and title.
    effect:
    """
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.plot(time_series, label = "Series")
    plt.plot(time_series.rolling(int(.05 * len(time_series))).mean(), '--', 
                label = "Rolling mean")
    plt.plot(time_series.rolling(int(.05 * len(time_series))).std(), ":",
                label = "Rolling Std")
    plt.title("Overview Description Plot of " + name.replace("_", " "))
    if name == "tsla_close":
        plt.ylabel("stock price")
    plt.xlabel("days")
    plt.legend(loc = "best")
    plt.savefig(name + "_description.png")
    plt.close()


def plot_decomposition(time_series, trend, seasonal, residual):
    """
    time_series: 
    trend:
    seasonal:
    residual:
    effect:
    """
    plt.figure(figsize=(12, 7))
    plt.subplot(411)
    plt.title("Seasonal Decomposition")
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


def plot_diff(time_series, copy_series, counts, name):
    """
    time_series: 
    copy_series:
    counts:
    name: used for later label and title.
    effect:
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label='Original', color='blue')
    plt.plot(copy_series, label='Diff' + str(counts), color='red')
    plt.title(name + " v.s. Time Series After Difference")
    plt.legend(loc='best')
    plt.savefig(name + "_diff.png")
    plt.close()


def plot_acf_pacf(time_series, name):
    """
    time_series: the time series which used to plot.
    name: used for later label and title
    effect: save the plot of acf and pacf.
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    plot_acf(time_series.tolist(), lags = 50, ax = ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(time_series.tolist(), lags = 50, ax = ax2)
    plt.title(name.replace("_", " ") + " ACF & PACF Plot")
    plt.savefig(name + "_acf_pacf.png")
    plt.close()

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


def model_predict(trend_arima_fit, residual_arima_fit, 
                  trend_garch_order, residual_garch_order,
                  trend, residual, seasonal, 
                  trend_diff_counts, residual_diff_counts, 
                  if_pred, start, end, period):
    """
    trend_arima_fit: ARIMA model after fit the trend.
    residual_arima_fit: ARIMA model after fit the residual.
    trend_garch_order: best parameters for GARCH model after fit the trend_arima_fit.resid.
    residual_garch_order: best parameters for GARCH model after fit the residual_arima_fit.resid.
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
        trend_pred_seq = np.array(trend_arima_fit.predict(start = date_after_train, 
                                                          end = end,
                                                          dynamic = True)) # The dynamic keyword affects in-sample prediction. 
        
        # get the residual predicted sequence from the start of start to end
        residual_pred_seq = np.array(residual_arima_fit.predict(start = date_after_train,
                                                                end = end,
                                                                dynamic = True))
        
        # find the the corresponding seasonal sequence.
        pred_period = (datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(date_after_train, '%Y-%m-%d %H:%M:%S')).days + 1
        
        trend_pred_variance, residual_pred_variance = np.zeros(pred_period), np.zeros(pred_period)
        current_trend_resid, current_residual_resid = trend_arima_fit.resid, residual_arima_fit.resid
        for i in range(pred_period):
            trend_model = arch_model(current_trend_resid,
                                     mean = "Constant",
                                     p = trend_garch_order[0], 
                                     q = trend_garch_order[1], 
                                     vol = 'GARCH')
            trend_model_fit = trend_model.fit(disp = "off",
                                              update_freq = 0,
                                              show_warning = False)
            trend_pred_variance[i] = np.sqrt(trend_model_fit.forecast(horizon = 1).variance.values[-1,:][0]) + trend_model_fit.forecast(horizon = 1).mean.values[-1,:][0]
            current_trend_resid.append(pd.DataFrame.from_dict({current_trend_resid.index.tolist()[-1] + relativedelta(days= 1): trend_pred_variance[i]}, orient = "index"))

            residual_model = arch_model(current_residual_resid,
                                        mean = "Constant",
                                        p = residual_garch_order[0], 
                                        q = residual_garch_order[1], 
                                        vol = 'GARCH')
            residual_model_fit = residual_model.fit(disp = "off",
                                                    update_freq = 0,
                                                    show_warning = False)
            residual_pred_variance[i] = np.sqrt(residual_model_fit.forecast(horizon = 1).variance.values[-1,:][0]) + residual_model_fit.forecast(horizon = 1).mean.values[-1,:][0]
            current_residual_resid.append(pd.DataFrame.from_dict({current_residual_resid.index.tolist()[-1] + relativedelta(days= 1): residual_pred_variance[i]}, orient = "index"))

        trend_pred_seq = trend_pred_seq + trend_pred_variance 
        residual_pred_seq = residual_pred_seq + residual_pred_variance

        trend_pred_seq = np.array(np.concatenate((np.array(trend.diff(trend_diff_counts).fillna(0)),
                                                  trend_pred_seq)))
        residual_pred_seq = np.array(np.concatenate((np.array(residual.diff(residual_diff_counts).fillna(0)),
                                                     residual_pred_seq)))
        seasonal_pred_seq = list(seasonal[len(seasonal) - period:]) * (round((pred_period) / period) + 1) 
        seasonal_pred_seq = np.array(seasonal_pred_seq[0:pred_period])
    else:
        trend_pred_seq = np.array(trend_arima_fit.fittedvalues)
        residual_pred_seq = np.array(residual_arima_fit.fittedvalues)
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


def GARCH_model(resid, args, name):
    """
    resid: stationary residual time_series after ARIMA. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return best parameters for GARCH model.
    """
    if args.plot:
        # plot acf and pacf for stationary time series
        # if we can find some autocorrelations from the graph, then we should use GARCH.
        plot_acf_pacf(resid, "GARCH_" + name + "_resid")

    best_criteria = np.inf 
    best_model_order = (0, 0)
    best_model_fit = None
    for p in range(args.max_p):
        for q in range(args.max_q):
            try:
                model = arch_model(resid,
                                   mean = "Constant",
                                   p = p, 
                                   q = q, 
                                   vol = 'GARCH')
                model_fit = model.fit(disp = "off",
                                      update_freq = 0,
                                      tol = args.tol,
                                      show_warning = False)

                if args.ic == "aic":
                    current_criteria = model_fit.aic
                else:
                    current_criteria = model_fit.bic
                
                if current_criteria <= best_criteria:
                    best_criteria, best_model_order, best_model_fit = np.round(current_criteria, 0), (p, q), model_fit
            except:
                logger.warning("Error occurs, try another combination of p and q.")
                pass

    return best_model_fit, best_model_order


def ARIMA_model(time_series_diff, args, name):
    """
    time_series_diff: stationary time_series after diff. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return fitted ARIMA model, parameters for ARIMA model.
    """
    if args.plot:
        # plot acf and pacf for stationary time series
        plot_acf_pacf(time_series_diff, "ARIMA_" +  str(name) + "_diff")

    # find the optimal order of ARIMA model.
    evaluate = sm.tsa.arma_order_select_ic(time_series_diff,
                                           ic = args.ic,
                                           trend = "c",
                                           max_ar = args.max_ar,
                                           max_ma = args.max_ma)
    min_order = evaluate[args.ic + "_min_order"] # get the parameter for ARIMA model.

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


def mix_model(time_series_diff, args, name):
    """
    time_series_diff: stationary time_series after diff. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return fitted ARIMA model, parameters for ARIMA model, fitted GARCH model and parameters for GARCH model.
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

    # get arima model
    arima_model_fit, arima_order = ARIMA_model(time_series_diff, args, name)

    if args.plot:
        # residual plots of residual model
        plot_residual(arima_model_fit.resid, name)

    # check if the resid of arima model is white noise
    _, pvalue = acorr_ljungbox(arima_model_fit.resid,
                               # auto_lag=True,
                               model_df=sum(arima_order),
                               return_df=False
                               )
    logger.info("acorr_ljungbox: " + str(list(pvalue)))
    if args.plot:
        plot_pvalue(pvalue, "acorr_ljungbox")
    if np.sum(pvalue < 0.05) > 0:
        logger.info("residual after fit still can not give white noises, we turn to use GARCH")
    else:
        logger.info("Although the residual does give good random values, we still turn to GARCH")
    
    # get garch model
    garch_model_fit, garch_order = GARCH_model(arima_model_fit.resid, args, name)

    return arima_model_fit, arima_order, garch_model_fit, garch_order


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
    while ADF(copy_series.tolist())[1] > 0.05:
        logger.info("time " + str(counts) + " ADF test: " + str(ADF(copy_series.tolist())))
        copy_series = copy_series.diff(1)
        copy_series = copy_series.fillna(0)
        counts += 1
    
    logger.info("time " + str(counts) + " ADF test: " + str(ADF(copy_series.tolist())))

    # plot diff and original time series in one graph.
    if if_plot:
        plot_diff(time_series, copy_series, counts, name)
    
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
        plot_decomposition(time_series, trend, seasonal, residual)

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
    parser = argparse.ArgumentParser(description="TSLA-ARIMA-GARCH")

    # hyperparameters setting
    parser.add_argument('--month', type=int, default=4,
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

    parser.add_argument('--max_p', type=int, default=4,
                        help='Maximum lag order of the symmetric innovation. Default 4.')

    parser.add_argument('--max_q', type=int, default=4,
                        help='Maximum lag order of lagged volatility or equivalent. Default 4.')

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
        logger.warning("The data range is illegal. Turn to use default 4")
        args.month = 4
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

    # data analyzing.
    logger.info("ADF test for tsla Close: " + str(ADF(tsla_close.tolist())[1]))
    # plot the graph describe tsla close
    if args.plot:
        plot_description(tsla_close, "tesla_close")

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

    # EDA of decomposed data, trend and residual
    if args.plot:
        plot_description(trend, "trend")
        plot_description(residual, "residual")

    # difference
    logger.debug("-----------------Diff-----------------")
    trend_diff, trend_diff_counts = diff(trend, args.plot, "trend", args.diff)
    logger.debug("trend diff counts: " + str(trend_diff_counts))
    residual_diff, residual_diff_counts = diff(residual, args.plot, "residual", args.diff)
    logger.debug("residual diff counts: " + str(residual_diff_counts))
    
    # ARIMA model
    logger.info("-------ARIMA GARCH construction--------")
    trend_arima_fit, trend_arima_order, trend_garch_fit, trend_garch_order = mix_model(trend_diff, args, "trend")
    logger.info("Trend ARIMA parameters: " + str(tuple([trend_arima_order[0],
                                                        trend_diff_counts,
                                                        trend_arima_order[1]])))
    logger.info("Trend GARCH parameters: " + str(trend_garch_order))
    residual_arima_fit, residual_arima_order, residual_garch_fit, residual_garch_order = mix_model(residual_diff, args, "residual")
    logger.info("Residual ARIMA parameters: " + str(tuple([residual_arima_order[0],
                                                          residual_diff_counts,
                                                          residual_arima_order[1]])))
    logger.info("Residual GARCH parameters: " + str(residual_garch_order))

    # model summary
    logger.debug("---------trend ARIMA summary----------")
    logger.debug(trend_arima_fit.summary())
    logger.debug("---------resid ARIMA summary----------")
    logger.debug(residual_arima_fit.summary())
    logger.debug("---------trend ARIMA summary----------")
    logger.debug(trend_garch_fit.summary())
    logger.debug("---------resid GARCH summary----------")
    logger.debug(residual_garch_fit.summary())

    logger.debug("-----trend model residual describe----")
    logger.debug(trend_arima_fit.resid.describe()) # describe the dataframe 
    logger.debug("-----resid model residual describe----")
    logger.debug(residual_arima_fit.resid.describe()) # describe the dataframe

    # loss calculation
    logger.info("-----------Loss calculation------------")
    fit_seq = model_predict(trend_arima_fit, residual_arima_fit,
                            trend_garch_order, residual_garch_order,
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
        plot_pred_vs_truth(fit_seq, np.array(train), "fit", "train")

    if list(test):
        pred_seq = model_predict(trend_arima_fit, residual_arima_fit,
                                trend_garch_order, residual_garch_order,
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
            plot_pred_vs_truth(pred_seq, np.array(test), "pred", "test")

    # prediction
    logger.info("--------------prediction---------------")
    prediction = model_predict(trend_arima_fit, residual_arima_fit,
                               trend_garch_order, residual_garch_order,
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

    logger.info("--------------File write---------------")
    with open("group_7.txt", "w+") as f:
        for pred in list(prediction):
            f.write("{:.3f}\n".format(pred))
    logger.info("--------------Process ends-------------")

if __name__ == "__main__":
    main()
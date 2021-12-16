# AppliedDeepLearning

This repository is aimed to show the progress of my project for the [Applied Deep Learning](https://tiss.tuwien.ac.at/course/educationDetails.xhtml?dswid=9806&dsrid=354&courseNr=194077&semester=2021W) course at TU Wien. 

*Disclaimer 1: Please note that this is a **work-in-progress** repository and therefore some details in this README may be updated by the end of the project.*

*Disclaimer 2: Please note that the whole content of **this repository is not aimed at giving any sort financial advice.** The focus is to explore the power of DL approaches on finance. What is more, no real money will be used for testing the proposed strategy.*

# Assignment 1: Project Overview

The goal of this project is to **implement a trading bot for a subset of S&P 500 stocks.** The project is splited in two clearly defined phases:

* Design, implementation, training, evaluation and comparison of prediction models.
* Design, implementation and automatization of a (customizable) traiding strategy.

## Prediction Models

As in many other approaches of stock market preidciton [[1]](#1) [[2]](#2), I will focus on **predicting the direction of the stocks' adjusted close price.** That is, the expected increase or decrease of a particular stock within a given time stamp (i.e., in 5 days from current time stamp). The reason why I seek to predict the gradient of the prices rather than the prices themselves is that: 1) most trading strategies are based on this concept and 2) predicting the price with an acceptable error may be easy (due to the usual high correlation with previous time stamp price), but actually useless for actual trading, as pointed out in this very informative [article](https://towardsdatascience.com/stock-prediction-using-recurrent-neural-networks-c03637437578). On the other hand, adjusted close prices are used rather than raw close prices since they reflect a more accurate version of the actual stock price by taking into account corporate actions such as dividends or right offerings, which affect trading's return. A detailed explanation of adjusted close prices can be found in this [article](https://www.investopedia.com/terms/a/adjusted_closing_price.asp).


## Dataset 

The data to be used in this project has been extracted using the [AlphaVantage API](https://www.alphavantage.co/) which is a widely used free API for financial market data extraction. (See [Load_API_Data.py](Load_API_Data.py)) 

Some details on the extracted data:

* API call function: `time_series_daily_adjusted()` which return raw **daily** open/high/low/close/volume values, **daily** adjusted close values and historical split/dividend events of the gloabl equity specified, covering 20+ years of historical data. 
* Stocks: as mentioned, only a subset of stocks are selected for this project. The process of choosing which stocks to be used consisted on several steps;
    * Retrieving historical data of the fifty *S&P 500* stocks with highest capitalization.
    * Plot an interactive visualization of those stocks in the period 2010-2020 (training and validation periods - see later). Feel free to play with the visualizations in the *uncommented* [plotAdjustedPrices.py](trash&tests\plotAdjustedPrices.py) script.
    * According to those visualizations, selection of a subset of them based on a subjective criteria to have as much variability on the data as possible (high volatility, frequency of peaks and valleys, etc.).
    * Selected stocks: **['AAPL', 'TSLA', 'AMGN'].** For a reference of the symbols, please check the [list of symbols for New York Stock Exchange.](https://eoddata.com/symbols.aspx?AspxAutoDetectCookieSupport=1)

<br>

![Google adjusted close price in the period 2012-2020](images/GOOGLE_ADJ_CLOSE_2012_2020.png)

<center>
Figure 1: Google adjusted close price in the period 2012-2020.
</center>

<br>


After running the function saveData() in [Load_API_Data.py](Load_API_Data.py) script, for each stock, a .csv file like [this one](data\AAPL.csv) is stored in the data folder. This file is minimally preprocessed (renaming and reindexing) when loaded for next steps using the function loadData() from the same script [Load_API_Data.py](Load_API_Data.py). All in all, the raw data of each stock looks as follows: 

<center>

|date      |open    |high    |low     |close     |adjustedClose |volume     |dividendAmount|SplitCoefficient|
|----------|--------|--------|--------|----------|--------------|-----------|--------------|----------------|
|1999-11-01|80.0    |80.69   |77.37   |77.62     |0.594976289312|2487300.0  |0.0           |1.0             |
|1999-11-02|78.0    |81.69   |77.31   |80.25     |0.615135882727|3564600.0  |0.0           |1.0             |
|1999-11-03|81.62   |83.25   |81.0    |81.5      |0.624717438533|2932700.0  |0.0           |1.0             |
|1999-11-04|82.06   |85.37   |80.62   |83.62     |0.640967757179|3384700.0  |0.0           |1.0             |
|1999-11-05|84.62   |88.37   |84.0    |88.31     |0.676917754562|3721500.0  |0.0           |1.0             |

Table 1: 5 samples from raw APPL daily data. 

</center>


For a proper training, validation and evaluation of the models, the data was **split in 3 sets** (per stock) as follows:

<center>

|split     |begin date|end date  |samples|purpose                                                     |
|----------|----------|----------|-------|------------------------------------------------------------|
|train     |01/01/2010|31/12/2019|2556 (trading days in the period 2010-2019)  |training models                                                   |
|validation|01/01/2020|31/12/2020|253 (trading days 2020)   |model selection, feature selection and hyperparameter tuning|
|test      |01/01/2021|31/12/2021|252 (trading days 2021)   |out-of-sample evaluation at the very end of the project - report and presentation     |


Table 2: Train, validation and test splits.

</center>


A similar train-val-test split can be found in other stock prediction works such as [[4]](#4)

## Target

As pointed out at the beginning of this section, the idea is to predict the direction of the stock price of several stocks (usually expressed as an ∆ in percentage points). However, rather than predicting the gradient itself, I will approach this problem as a **classification problem**, in which I will try to classify the future price of a stock on given ∆% interval bins. More concretly, I will focus on two classification tasks (**multitarget**): 

* Target 1 classes:
    * Class 0: [-∞, 0%] - (i.e, price decreases)
    * Class 1: [0%, ∞] - (i.e, price increases)

<br>

* Target 2 classes: 
    * Class 0:  [-∞, -1.5%]
    * Class 1:  [-1.5%, 1.5%]
    * Class 2:  [1,5%, ∞]

The idea is to test, on the one hand, the ability of the model on the most simple prediction of the price direction (target 1) and, on the other hand, the ability of the model on finding more profitable trading opportunities in which the price increases or decreases significantly (classes 0 and 2 in target 2). Likely, the second version will be more challenging due to the fact that it has more classes and that we can expect a class imbalance favoring predictions of the middle interval (class 1). 

So far, the concrete time stamp for predictions has not been defined (i.e., whether I want to predict tomorrow's price or next week's). The idea is in fact to test 3 options along this project: 

* Predict **tomorrow's** adjusted close price (+1 trading day from current time stamp)
* Predict **next week's** adjusted close price (+5 trading days from current time stamp)
* Predict **next next week's** adjusted close price (+10 trading days from current time stamp)

This way will allow me to experiment on the performance of the trained models for multiple time stamp predictions and targets. In the end, I would like to select 6 models (2*3) - per stock - and compare the performance across classes, prediction time stamps and stocks. 
<br>

*Note that the scope may be reduced/extended depending on training times & other issues. The idea is to implement a flexible framework to allow me to test several intervals & prediction time stamps.*

## Preprocessing

Extraction of features to experiment on their effect on performance. Feature types to be extracted:

* Lagged prices: adjusted closed prices from the previous days. This is a common type of feature used in most of stock prediction models. 

* Technical indicators: mathematical calculations performed on different stock parameters historically used by traders to implement their strategies.
    * Type of indicators: overlap studies, momentum indicators, volume indicators, etc. 
    * Library to be used: [TA-Lib](https://mrjbq7.github.io/ta-lib/doc_index.html). 

* *Potentially:* Finacial News; perform some kind of classification for financial news (i.e., positive vs. negative) of each of the stocks selected and use them as an additional feature for prediction.

* *Potentially:* Related Stock Prices; include the price of other stocks related to the target as an additional predictor (i.e., those showing a strong negative correlation with respect to target).


Feature transformation/selection: 

* Principal Component Analysis (PCA)
* SelectKBest

The idea is to dynamically experiment on which features bring the best performance, whehter reducing their dimensionality eases prediction, etc. 

Table 3 shows and example of how a preprocessed dataset could look like. 

<center>

|date      |open      |high      |low |close                                                       |adjustedClose|volume    |dividendAmount|SplitCoefficient|close_lag_1  |close_lag_3  |close_lag_5  |AROONOSC_Days_5|AROONOSC_Days_10|AROONOSC_Days_14|AROONOSC_Days_20|AROONOSC_Days_25|targetadjustedCloseIncrease1Days_[-inf, -3, -2, -1, 0, 1, 2, 3, inf]|targetadjustedCloseIncrease1Days_[-inf, -4, 4, inf]|targetadjustedCloseIncrease1Days_[-inf, 0, inf]|targetadjustedCloseIncrease1Days_[-inf, -1, 1.2, inf]|targetadjustedCloseIncrease2Days_[-inf, -3, -2, -1, 0, 1, 2, 3, inf]|targetadjustedCloseIncrease2Days_[-inf, -4, 4, inf]|targetadjustedCloseIncrease2Days_[-inf, 0, inf]|targetadjustedCloseIncrease2Days_[-inf, -1, 1.2, inf]|targetadjustedCloseIncrease3Days_[-inf, -3, -2, -1, 0, 1, 2, 3, inf]|targetadjustedCloseIncrease3Days_[-inf, -4, 4, inf]|targetadjustedCloseIncrease3Days_[-inf, 0, inf]|targetadjustedCloseIncrease3Days_[-inf, -1, 1.2, inf]|targetadjustedCloseIncrease5Days_[-inf, -3, -2, -1, 0, 1, 2, 3, inf]|targetadjustedCloseIncrease5Days_[-inf, -4, 4, inf]|targetadjustedCloseIncrease5Days_[-inf, 0, inf]|targetadjustedCloseIncrease5Days_[-inf, -1, 1.2, inf]|targetadjustedCloseIncrease10Days_[-inf, -3, -2, -1, 0, 1, 2, 3, inf]|targetadjustedCloseIncrease10Days_[-inf, -4, 4, inf]|targetadjustedCloseIncrease10Days_[-inf, 0, inf]|targetadjustedCloseIncrease10Days_[-inf, -1, 1.2, inf]|targetadjustedCloseIncrease15Days_[-inf, -3, -2, -1, 0, 1, 2, 3, inf]|targetadjustedCloseIncrease15Days_[-inf, -4, 4, inf]|targetadjustedCloseIncrease15Days_[-inf, 0, inf]|targetadjustedCloseIncrease15Days_[-inf, -1, 1.2, inf]|
|----------|----------|----------|----|------------------------------------------------------------|-------------|----------|--------------|----------------|-------------|-------------|-------------|---------------|----------------|----------------|----------------|----------------|--------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------|------------------------------------------------------|
|2012-01-03|409.4     |412.5     |409.0|411.23                                                      |12.6087142208|10793600.0|0.0           |1.0             |12.4176963242|12.3453364148|12.36649249  |100.0          |100.0           |85.71428571428572|60.0            |100.0           |6                                                                   |2                                                  |2                                              |3                                                    |6                                                                   |2                                                  |2                                              |3                                                    |7                                                                   |2                                                  |2                                              |3                                                    |6                                                                   |2                                                  |2                                              |3                                                    |8                                                                    |3                                                   |2                                               |3                                                     |8                                                                    |3                                                   |2                                               |3                                                     |
|2012-01-04|410.0     |414.68    |409.28|413.44                                                      |12.6764749834|9286500.0 |0.0           |1.0             |12.6087142208|12.4213756416|12.4646076214|60.0           |100.0           |92.85714285714286|65.0            |96.0            |5                                                                   |2                                                  |2                                              |2                                                    |7                                                                   |2                                                  |2                                              |3                                                    |7                                                                   |2                                                  |2                                              |3                                                    |6                                                                   |2                                                  |2                                              |3                                                    |8                                                                    |3                                                   |2                                               |3                                                     |8                                                                    |3                                                   |2                                               |3                                                     |


Table 3: sample of a potential preprocessed dataset of AAPL stock (prior to X-Y split). 

</center>


## Models

In this section I plan to train a few traditional classificaction algorithms (such as Random Forest Classifier, Support Vector Classifier, AdaBoost, etc.) which will establish a baseline to be compared with the performance of several DL approaches. As for the DL approaches, I will focus on LSTM nets, which have been proven to work good for time-series prediction. More concretly, I will inspire my approach on the structure proposed in [[5]](#5). If time allows for it, I would like to experiment with including *attention* in my models too. 

*More detailed information on this section will be provided as the project advances.*

## Evaluation

Since the problem will be approached as a classification task, I will use **classification evaluatuion metrics** for the model evaluation. Therefore, which model to select for the automatized trading strategy will be decide based on a deep analysis of validation accuracy, recall, precision, etc., depending on the specific needs of the trading strategy. 

*More detailed information on this section will be provided as the project advances.*

## Automatized Trading Strategy

This is so far the least clear section of the project so far. 

On the one hand,, I plan to design a simulator in which a potential user can **backtest** the strategy by selecting how much to invest, in which time period, on which stocks, etc. and return the results summarized. A user could choose for instance to invest 1000€ on Apple from 2014 to 2018 based on the implemented strategy. Once the simulation is run, a summary with the return, alpha value, risk, etc. will be shown to the user. Which concrete trading strategy to use will depend on the capacity of the model to predict the different targets, however, some version of the Martingale could be a good beginning to test the potential of the experiments. 

On the other hand, I plan to design a trading bot that makes decision based on the model's predictions. To do so, I would like to automatize a simple script that should be run everyday after stock market closes. This method will query the API for latest price information of the selected stocks, predict the direction of the price movement and design a trading order based on those predictions to be implemented once the market reopens again. Of course, all of this using paper trading (fake money). For this second part of the application, I will use the free paper trading platform [InteractiveBrokers](https://interactivebrokers.github.io/tws-api/introduction.html). This second part of the application is aimed to be tested during the days between the test period ends (31/12/2021) to the day of the presentation (19/01/2022).


## Project Type

As described in the "Model" section, the idea is to experiment on the capacity of modern deep learning nets (like LTSM) to beat traditional machine learning approaches (such RF classifier) on stock movement prediction (i.e., **beat the classics**). However, I would also like to experiment, as described, with different architectures and customize my own method for this task. Therefore, the project could also partially fit on "Bring your own method" category.

## Work-breakdown structure (Approximation)

* Reading biblography, figuring out the idea, setting up the repository and writing README: 20 hours. 
* Data Extraction, Target Definition & Flexible Preprocessing Pipeline: 7 hours
* Training traditional ML models and storing results: 10 hours (implementation) + training time
* Training DL models and storing results: 25 hours (implementation) + training time
* Analyzing results, fine-tuning, re-training and experimenting new approaches: 25 hours + training time
* Design and implementation of the trading strategy: 30 hours
* Writing final report: 6 hours
* Preparing presentation: 3 hours 


## References

<a id="1">[1]</a> 
Matsubara, T., Akita, R., & Uehara, K. (2018). Stock price prediction by deep neural generative model of news articles. IEICE TRANSACTIONS on Information and Systems, 101(4), 901-908.

<a id="2">[2]</a> 
Ghosh, P., Neufeld, A., & Sahoo, J. K. (2021). Forecasting directional movements of stock prices for intraday trading using LSTM and random forests. Finance Research Letters, 102280.

<a id="3">[3]</a> 
Matsubara, T., Akita, R., & Uehara, K. (2018). Stock price prediction by deep neural generative model of news articles. IEICE TRANSACTIONS on Information and Systems, 101(4), 901-908.

<a id="4">[4]</a> 
Zhang, L., Aggarwal, C., & Qi, G. J. (2017, August). Stock price prediction via discovering multi-frequency trading patterns. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 2141-2149).

<a id="5">[5]</a> 
Mehtab, S., Sen, J., & Dutta, A. (2020, October). Stock price prediction using machine learning and LSTM-based deep learning models. In Symposium on Machine Learning and Metaheuristics Algorithms, and Applications (pp. 88-106). Springer, Singapore.


# Assignment 2: Hacking

## Error metric


- Accuracy (for only up and down)
- Precision: we want to be sure that when a big increase is predicted, this is actually the case (specially for the second target). 

## Target error metric

- Better than predicting majority class only
- More than random prediction on test
- Threshold defined 
- Emergency threshold defined 
- 2% monthly increase strategy + beat alpha of the market

## Actually achieved value

- Train: 
- Validation: 
- Test: 

## Amount of time

- 

## Explain code

- 

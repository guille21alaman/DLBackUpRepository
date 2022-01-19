#Package imports
from __future__ import annotations
from calendar import weekday
from email.policy import default
from turtle import color, width
import streamlit as st
import pandas as pd
import numpy as np 
import os
from joblib import dump, load
from datetime import date, datetime
import math
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pyautogui



#imports for preprocessing
from MLPreprocessing import *


############# Title and description
st.write(
    """
    # Trading Simulator
    Welcome to this trading simulator!
    """
)

#image
front_image = Image.open("architectures/trading.jpg")
st.image(front_image)

#features
st.write("""
    #### Descriptive features of the simulator
    - Based on price direction prediction in **10 days** from the current day.
    - Model (AdaBoostClassifier) **trained on AMGN historical data** (2013-2019) as shown in previous assignments. 
    - Trading strategy: buy when increase predicted and sell after 10 days (**10 independent lines of investment**) - could be very improved.
    - Evaluation of the trading strategy: **investment return is compared with market return (alpha)** and random return.
    - Simulation is allowed in different datasets and date ranges as shwon in the section *simulator parameters*.
    - **Balloons** are shown **if** the tested strategy returns a positve **investment above market return.**
""")



############ Inputs
#region
st.sidebar.markdown("## Simulator Parameters")

## reset button 
if st.sidebar.button("Reset"):
    pyautogui.hotkey("ctrl","F5")

#extract available stocks for the simulator
stocks = []
for s in os.listdir("data/"):
    stocks.append(s[:-4])

#slider for selection
stock = st.sidebar.selectbox(label="Please, select a stock for simulation: ",
    #list of options to be selected
    options = stocks,
    #preselected option is AMGN
    index = 5, 
    help="The selected stock will be used in the simulation of the upcoming sections. Only top 50 capitalization companies from S&P 500 available."
    )

#set condition that a stock should be chosen
if stock == "-":
    st.sidebar.error("A stock must be chosen to proceed with the simulation.")

#load dataframe of selected stock (if not error)
else:
    df = pd.read_csv("data/" + stock+".csv", sep=";", decimal=",")

    #get minimum and maximum date for each loaded datasets (to be used later)
    minimum_date = datetime.datetime.strptime(df["date"][df.shape[0]-1], '%Y-%m-%d')
    maximum_date = datetime.datetime.strptime(df["date"][0], "%Y-%m-%d") #date until we have data

    #columns for renaming
    columns = ["open", "high", "low", "close", "adjustedClose", "volume",
     "dividendAmount", "SplitCoefficient"]

    #rename columns (using columns list taken as an argument)
    #hard-coded because the output data from AlphaVantage Follows always the same structure
    df = df.rename(columns={
        "1. open": columns[0],
        "2. high": columns[1],
        "3. low": columns[2],
        "4. close": columns[3],
        "5. adjusted close": columns[4],
        "6. volume": columns[5],
        "7. dividend amount": columns[6],
        "8. split coefficient": columns[7]
    })

    #make date index
    df = df.set_index("date")
    #convert to datetime (double check)
    df.index = pd.to_datetime(df.index)
    #sort values by dates (ascending)
    df = df.sort_values(by="date", ascending = True)

    

#preload pre-trained model
#load trained model on AMGN data 
model_path = "results/AMGN/models/1_10[-inf,0,inf]_AdaBoostClassifier__StandardScaler_SelectKBest_cv_default__model__learning_rate_1.joblib"
model = load(model_path)
column_name = "targetadjustedCloseIncrease10Days_[-inf, 0, inf]"


#correct flag for data
correct = False

# Choose dates for simulation

#start date selection (Choose only weekdays and non-holidays)
start_date = st.sidebar.date_input(
    label="Choose the date range in which you would like to simulate our 10-day prediction algorithm. \n Select a start date: ",
    value = minimum_date + datetime.timedelta(365*2+2), #min value needs to be above a threshold to allow for a proper preprocessing (at least 2 year margin)
    min_value= minimum_date + datetime.timedelta(365*2+2),
    max_value= maximum_date - datetime.timedelta(20)) #max value enough to compute ranges

trading_days = st.sidebar.number_input(
    label="How many days would you like to simulate (default one year - note that trading days are actually ~252 per year). If days exceed maximum, then take until last entry on the evaluation set.",
    value = 365,
    min_value=1,
    max_value=7300 #there are only 20 years of data (as maximum)
)

if trading_days <1:
    st.sidebar.error("Enter a positive number.")

#add desired trading years and correct for 10 days prediction (14 days taking into account weekends)
end_date = start_date + datetime.timedelta(trading_days) + datetime.timedelta(14)

#transform those to list to fit function
start_date_list = [start_date.day, start_date.month, start_date.year]
end_date_list = [end_date.day, end_date.month, end_date.year]


# Preprocess data according to dates
#params and function call 
#features to be used
target = "adjustedClose" 
main_columns = ["open", "high", "low", "close", "adjustedClose", "volume", "dividendAmount", "SplitCoefficient"] 

#shift days for the target increase
shiftDays = [1,5,10]

#for the prices to use as predictors
lags = [1,2,3,4,5,6,7,8,9,10,20,50]

#technical indicators
indicators = ["BBANDS", "HT_TRENDLINE", "MA",
 "SMA", "DEMA", "KAMA", "SMA", "T3", "TEMA", "TRIMA", "WMA",
 "DX", "AROOSC", "BOP", "EMA"]

#intervals for the targets
intervals = [
    [-math.inf, -1.5, 1.5, math.inf], # catch increases that are worthy
    [-math.inf, 0, math.inf] #up or down   
]

#dict to have right input for fuction
df_dict = {}
df_dict[stock] = df

train, trash = preprocess(df_dict, main_columns, start_date_list,
 end_date_list, start_date_list, end_date_list, shiftDays,
 indicators, intervals, lags, target)

#save as dataframe and get date index of selected set
df_simulation = train[stock]
df_simulation_index = df_simulation.index

#split data 
X, Y = X_Y_split(train) 
X_simulation = X[stock]
Y_simulation = Y[stock]


#prepare dataset for checking with simulation (i.e. same as simulation + 10 days)
#get index first simulation day
index_firstDay_df_simulation = df_simulation.index[0]
#match this index in general dataframe and take from next 10 rows (and until the length of df simulation + 10)
df_test_simulation = df.loc[index_firstDay_df_simulation:].iloc[10:df_simulation.shape[0]+10]


#total investment to be used
investment = st.sidebar.number_input(
    "Enter the amount of money you desire to invest: ",
    min_value=0,
    value=10000
)

st.markdown("""---""")

st.write("""
    #### Data Description
""" )


col1,col2 = st.columns(2)

col1.write("""
    **Price evolution** on the chosen period: 
    - **Stock: ** **%s**
    - **Trading period: ** (%s to %s)
    - **Total initial investment: ** %s€
"""%(stock,df_simulation_index[0].strftime("%Y-%m-%d"),
 df_simulation_index[-1].strftime("%Y-%m-%d"), int(investment)))

sns.lineplot(df_test_simulation.iloc[:-10].index,
 df_test_simulation.iloc[:-10].adjustedClose, color="lightblue")
plt.xticks(rotation=45)
st.set_option('deprecation.showPyplotGlobalUse', False)
col1.pyplot()

col2.write("""
    **Descriptive statistics: **
    - **Average price**: %s€
    - **Minimum price**: %s€
    - **Maximum price**: %s€
    - **Standard deviation**: %s€ 
    - **Average volume traded:** %s€
"""%(
    round(df_test_simulation.iloc[:-10].adjustedClose.mean(),2),
    round(df_test_simulation.iloc[:-10].adjustedClose.min(),2),
    round(df_test_simulation.iloc[:-10].adjustedClose.max(),2),
    round(df_test_simulation.iloc[:-10].adjustedClose.std(),2),
    round(df_test_simulation.iloc[:-10].volume.mean(),2),

))


#endregion


############# Make predictions on price direction

#region

def print_evaluation(predictions, y_true):

    print("\n\nEvaluation Results: ")
    accuracy = accuracy_score(y_true, predictions)
    precision_macro = precision_score(y_true, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    print("- Accuracy score:\n ", accuracy)
    print("- Precision macro:\n", precision_macro)
    print("- Confusion Matrix:\n ", cm)

    return cm, accuracy, precision_macro


column_name = "targetadjustedCloseIncrease10Days_[-inf, 0, inf]"
print("\n\n\nModel: ", model)
print("\nTarget: ", column_name)


predictions = model.predict(X_simulation)
predictions = pd.Series(predictions)
cm, accuracy, precision = print_evaluation(Y_simulation[column_name], predictions)


#endregion

################## Make trading simulator

#store all the information so far in a single dataframe
X_simulation["Ground_Truth"] = Y_simulation[column_name]
X_simulation["Predictions"] = predictions.values
X_simulation["AdjustedClose10Days"] = df_test_simulation["adjustedClose"].values
X_simulation["AccuracyPrediction"] = X_simulation["Predictions"] == X_simulation["Ground_Truth"]
X_simulation["RandomPredictions"] = np.random.randint(1,3, size=X_simulation.shape[0])

#select only needed columns
selected_columns = ["adjustedClose","AdjustedClose10Days","Ground_Truth","Predictions","AccuracyPrediction","RandomPredictions"]
X_simulation = X_simulation[selected_columns]
df_trading = X_simulation.copy()

st.markdown("""---""")

st.write("""
    #### Interpolation Results
""" )

#plot positive vs negative predictions
col1, col2 = st.columns(2)


col1.write("** Price colored by predictive success** on 10 day prediction:")
sns.scatterplot(df_trading.index, df_trading["adjustedClose"],
 s=30,
 hue=(df_trading["Predictions"] == df_trading["Ground_Truth"]),
 palette = ["salmon", "lightgreen"]
 )
plt.ylabel("Adjusted Close Price")
plt.xticks(rotation=45)
plt.legend(title="10 Day Price Direction\nPrediction Success",
 loc="upper right", labels=["Correct","Wrong"])
col1.pyplot()

col2.write("""
    **Summary statistics:**

    - **Overall Accuracy:** %s%%
    - **Overall Precision:** %s%%
    - **Precision price increase:** %s%%
    - **Precision price decrease:** %s%%
"""%(round(100*accuracy,2), round(100*precision,2),
    round(100*cm[0,0]/(cm[0,0]+cm[0,1]),2), round(100*cm[1,1]/(cm[1,0]+cm[1,1]),2)))


#Parameters


#Strategy: very simple; 
# If increase predicted: buy - sell after 10 days and check profit/losses. 
# If decrease predicted: ignore. 
# Since there is a time delay between investment and return --> the total money to be invested needs to be split (i.e., n independent investment strategies)


def trading_strategy(investment, df_trading, predictions):

    #start by dividing investment in 10 lines (as we have 10 days, so that we can trade every single day)
    investment_lines = {}
    for l in range(10):
        investment_lines[l] = investment/10

    #keep track of successful investments
    investments_success = {}
    investments_success["successful"] = {}
    investments_success["unsuccessful"] = {}
    count_successful = 0
    profit_successful = []
    return_successful = []
    count_unsuccessful = 0
    profit_unsuccessful = []
    return_unsuccessful = [] 

    #log dataframe to keep track of everything
    log_df = pd.DataFrame()

    current_investment_line = 0
    
    largest_win = 0
    largest_loss = 0
    large_moves = {}
    large_moves["win"] = {}
    large_moves["loss"] = {}

    for index, row in df_trading.iterrows():

        new_row = {}
    
        new_row["date"] = index
        new_row["priceToday"] = row["adjustedClose"]
        new_row["price10Days"] = row["AdjustedClose10Days"]


        #if predicted increase, buy & update current investment line
        if row[predictions] == 2:

            new_row["invested"] = investment_lines[current_investment_line]
            
            #compute how much is bought with current money in current investment line
            quantity_bought = investment_lines[current_investment_line]/row["adjustedClose"]

            new_row["quantityBought"] = quantity_bought

            #profit: quantity bought * price in 10 days (for when prediction was made)
            profit = quantity_bought*row["AdjustedClose10Days"]

            new_row["profit"] = profit
 
            #operation return
            operation_return = (profit - investment_lines[current_investment_line])/investment_lines[current_investment_line]

            new_row["return"] = operation_return

            if operation_return > 0:
                count_successful+=1
                profit_successful.append(profit-investment_lines[current_investment_line])
                return_successful.append(operation_return)
            else:
                count_unsuccessful+=1
                profit_unsuccessful.append(profit-investment_lines[current_investment_line])
                return_unsuccessful.append(operation_return)

            if operation_return > largest_win:
                largest_win = operation_return
                large_moves["win"]["closeToday"] = round(row["adjustedClose"],2)
                large_moves["win"]["close10Days"] = round(row["AdjustedClose10Days"],2)
                large_moves["win"]["previousInvestment"] = round(investment_lines[current_investment_line],2)
                large_moves["win"]["quantityBought"] = round(quantity_bought,2)
                large_moves["win"]["nextInvestment"] = round(profit,2)
                large_moves["win"]["return"] = round(operation_return,4)*100
                large_moves["win"]["date"] = index

            elif operation_return < largest_loss:
                largest_loss = operation_return
                large_moves["loss"]["closeToday"] = round(row["adjustedClose"],2)
                large_moves["loss"]["close10Days"] = round(row["AdjustedClose10Days"],2)
                large_moves["loss"]["previousInvestment"] = round(investment_lines[current_investment_line],2)
                large_moves["loss"]["quantityBought"] = round(quantity_bought,2)
                large_moves["loss"]["nextInvestment"] = round(profit,2)
                large_moves["loss"]["return"] = round(operation_return,4)*100
                large_moves["loss"]["date"] = index

            #update value of current investment line
            investment_lines[current_investment_line] = profit 

            #update current investment line (make sure of not unrealistic investments)
            if current_investment_line != 9:
                current_investment_line += 1
            else:
                current_investment_line = 0
            
            log_df = log_df.append(new_row, ignore_index=True)

        #else continue to the next day
        else:
            continue
    
    #fill dictionary successful/unsuccessful
    investments_success["successful"]["Count"] = count_successful
    investments_success["successful"]["Profit"] = profit_successful
    investments_success["successful"]["Return"] = return_successful

    investments_success["unsuccessful"]["Count"] = count_unsuccessful
    investments_success["unsuccessful"]["Profit"] = profit_unsuccessful
    investments_success["unsuccessful"]["Return"] = return_unsuccessful

    return investment_lines, large_moves, investments_success, log_df

investment_lines, large_moves, investments_success, log_df = trading_strategy(investment, df_trading, "Predictions")

#compute investment return
investment_after = 0
for i in investment_lines:
    investment_after += investment_lines[i]
investment_return = (investment_after - investment)/investment
raw_ganancies = investment_after - investment

#compute market return (alpha)
quantity_bought_initially = investment/df_trading.iloc[0,0]
final_cash = quantity_bought_initially*df_trading.iloc[-1,0]
market_return = (final_cash-investment)/investment

# Compute random return 
investment_lines, trash, trash2, trash3 = trading_strategy(investment, df_trading, "RandomPredictions")
investment_after = 0
for i in investment_lines:
    investment_after += investment_lines[i]
random_return = (investment_after - investment)/investment

if (investment_return >= market_return) & (investment_return > random_return) & (investment_return > 0):
    st.balloons()

st.markdown("""---""")

st.write("""#### Simulator results""")

#two columns for returns
col1, col2 = st.columns(2)

#barplot with returns
list_returns = [100*round(investment_return,2), 100*round(market_return,2), 100*round(random_return,2)]
returns = pd.DataFrame()
returns["Return in %"] = list_returns
returns["Type of Return"] = ["Investment", "Market", "Random"]
returns["colors"] = ["lightblue","lightblue","lightblue"]

col1.write("**Comparison of returns:** ")
plt.bar(returns["Type of Return"], returns["Return in %"], color=returns["colors"])
plt.ylabel("Return in %")
col1.pyplot()

col1.write("""
    - **Investment return**: %s%%
    - **Market return**: %s%%
    - **Random return**: %s%%
    - **Total raw ganancies of investment return: ** %s€
"""%(round(100*investment_return,2), round(100*market_return,2), round(100*random_return,2),
 round(raw_ganancies,2)))



#count of successful strategies vs. not successful strategies
col2.write("**Successful vs. not successful** investments:")

#figure with successful vs unsuccessful investments
success = pd.DataFrame()
success["Type of Success"] = ["Successful","Unsuccessful"]
success["Count"] = [investments_success["successful"]["Count"],
 investments_success["unsuccessful"]["Count"]]
success["colors"] = ["lightgreen","salmon"]

plt.bar(success["Type of Success"], success["Count"], color=success["colors"])
plt.ylabel("Count of investments")
col2.pyplot()



col2.write("""
    ###### Successful investments:
    - **Count: ** %s
    - **Total profit:** %s€
    - **Average return:** %s%%
    - **Largest positive return:** %s%%
    ###### Unsuccessful investments
    - **Count: ** %s
    - **Total losses:** %s€
    - **Average return:** %s%%
    - **Largest negative return** %s%%

"""%(
    investments_success["successful"]["Count"],
    round(sum(investments_success["successful"]["Profit"]),2),
    round(np.array(investments_success["successful"]["Return"]).mean()*100,2),
    round(np.array(investments_success["successful"]["Return"]).max()*100,2),
    investments_success["unsuccessful"]["Count"],
    round(sum(investments_success["unsuccessful"]["Profit"]),2),
    round(np.array(investments_success["unsuccessful"]["Return"]).mean()*100,2),
    round(np.array(investments_success["unsuccessful"]["Return"]).min()*100,2),
))



st.markdown("""---""")

#Largest/Lowest return
df_trading_date = df_trading.reset_index()
#2 columns to display them
col1, col2  = st.columns(2)

#best return
col1.write("**Best Return on an investment:**")
#data
index_best = df_trading_date[df_trading_date.date == large_moves["win"]["date"]].index[0]
plot_best = df_trading_date.iloc[index_best:index_best+11]
buy_points = [plot_best.iloc[0]["date"], plot_best.iloc[0]["adjustedClose"]]
buy_points_df = pd.DataFrame(columns=["date","adjustedClose"], data=[buy_points])
sell_points = [plot_best.iloc[-1]["date"], plot_best.iloc[-1]["adjustedClose"]]
sell_points_df = pd.DataFrame(columns=["date","adjustedClose"], data=[sell_points])
#plots
sns.scatterplot(x="date",y="adjustedClose", data=buy_points_df, s=150, color="lightgreen")
plt.annotate("--> Buy", (buy_points[0]+datetime.timedelta(0.5),buy_points[1]), fontsize=12)
sns.scatterplot(x="date",y="adjustedClose", data=sell_points_df, s=150, color="lightgreen")
plt.annotate("Sell <--", (sell_points[0]-datetime.timedelta(2.3),sell_points[1]-0.1), fontsize=12)
sns.lineplot(data=plot_best,
     x="date", y="adjustedClose", linewidth = 3, color="lightgreen")
plt.xticks(rotation=45)
st.set_option('deprecation.showPyplotGlobalUse', False)
col1.pyplot()
col1.write("""
    - **Money invested:** %s€
    - **Quantity bought:** %s
    - **Bought** on %s at %s€. 
    - **Sold** on %s at %s€.
    - **Money after transcation: ** %s€
    - **Total profit:** %s€
    - **Return:** %s%%
"""%(
    large_moves["win"]["previousInvestment"],
    round(large_moves["win"]["previousInvestment"]/large_moves["win"]["closeToday"],2),
    large_moves["win"]["date"].strftime("%Y-%m-%d"), large_moves["win"]["closeToday"],
    sell_points[0].strftime("%Y-%m-%d"), large_moves["win"]["close10Days"],
    large_moves["win"]["nextInvestment"],
    round(large_moves["win"]["nextInvestment"] - large_moves["win"]["previousInvestment"],2),
    large_moves["win"]["return"]
))


#worst return
col2.write("**Worst Return on an investment:**")

#data
index_best = df_trading_date[df_trading_date.date == large_moves["loss"]["date"]].index[0]
plot_best = df_trading_date.iloc[index_best:index_best+11]
buy_points = [plot_best.iloc[0]["date"], plot_best.iloc[0]["adjustedClose"]]
buy_points_df = pd.DataFrame(columns=["date","adjustedClose"], data=[buy_points])
sell_points = [plot_best.iloc[-1]["date"], plot_best.iloc[-1]["adjustedClose"]]
sell_points_df = pd.DataFrame(columns=["date","adjustedClose"], data=[sell_points])
#plots
sns.scatterplot(x="date",y="adjustedClose", data=buy_points_df, s=150, color="salmon")
plt.annotate("--> Buy", (buy_points[0]+datetime.timedelta(0.5),buy_points[1]), fontsize=12)
sns.scatterplot(x="date",y="adjustedClose", data=sell_points_df, s=150, color="salmon")
plt.annotate("Sell <--", (sell_points[0]-datetime.timedelta(2.3),sell_points[1]-0.1), fontsize=12)
sns.lineplot(data=plot_best,
     x="date", y="adjustedClose", linewidth = 3, color="salmon")
plt.xticks(rotation=45)
st.set_option('deprecation.showPyplotGlobalUse', False)
col2.pyplot()
col2.write("""
    - **Money invested:** %s€
    - **Quantity bought:** %s
    - **Bought** on %s at %s€. 
    - **Sold** on %s at %s€.
    - **Money after transcation: ** %s€
    - **Total profit:** %s€
    - **Return:** %s%%
"""%(
    large_moves["loss"]["previousInvestment"],
    round(large_moves["loss"]["previousInvestment"]/large_moves["loss"]["closeToday"],2),
    large_moves["loss"]["date"].strftime("%Y-%m-%d"), large_moves["loss"]["closeToday"],
    sell_points[0].strftime("%Y-%m-%d"), large_moves["loss"]["close10Days"],
    large_moves["loss"]["nextInvestment"],
    round(large_moves["loss"]["nextInvestment"] - large_moves["loss"]["previousInvestment"],2),
    round(large_moves["loss"]["return"],2)
))


st.markdown("""---""")


st.write("***Simulator Log: ***")
st.write(log_df)

st.write("VIsualization of *simulator log:*")
sns.scatterplot(x="date", y="return", data=log_df,
 hue=log_df["return"]>0, palette=["salmon", "lightgreen"])
plt.axhline(y=0, color="lightgrey")
plt.legend(title="Return > 0",
 loc="upper left",
  )
plt.xticks(rotation=45)
st.pyplot()

#endregion



import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import math
# Read a CSV file into a DataFrame
farmingCompanies = pd.read_csv('Farm.csv') #Data can be found in public sources
fedfundsDF = pd.read_csv('FEDFUNDS.csv')
wheatDF = pd.read_csv('Wheat.csv')
farmingCompanies=farmingCompanies.dropna()
fedfundsDF=fedfundsDF.dropna()
wheatDF=wheatDF.dropna()
corDF=pd.DataFrame()
print(wheatDF)
def filter_and_drop(df, column_to_filter, startDate, endDate):
    # Convert the date column to datetime
    df[column_to_filter] = pd.to_datetime(df[column_to_filter])
    # Identify rows where the date is older than 2004 or older than 2023
    rows_to_drop = df[(df[column_to_filter] < startDate) | (df[column_to_filter] > endDate)].index
    df.drop(index=rows_to_drop, inplace=True)
    return df


#adding element to correlation matrix via function + resetting index and avoiding duplicated code
def add_to_cor(df, tempColumn, name):
    tempColumn = tempColumn.reset_index(drop=True)
    df[name]=tempColumn

######## Getting revenue over time as a graph in order to track anomalies easier
    
revenue_df = farmingCompanies[farmingCompanies['item_name'] == 'totalRevenue']
total_revenue = revenue_df.groupby('ticker')['value'].sum()
sorted_revenue = total_revenue.sort_values(ascending=False)
top_5_tickers = sorted_revenue.head(5).index.tolist()

revenue_df['interval_date'] = pd.to_datetime(revenue_df['interval_date'])
fedfundsDF['DATE'] = pd.to_datetime(fedfundsDF['DATE'], infer_datetime_format=True)
revenue_df = revenue_df.sort_values('interval_date')
tickers = revenue_df['ticker'].unique()
colors = cm.get_cmap('tab20', len(tickers))
plt.figure(figsize=(14, 8))
fedfundsDF['FEDFUNDS_PERC']=(fedfundsDF['FEDFUNDS'] / fedfundsDF['FEDFUNDS'].max()) * 100
plt.plot(fedfundsDF['DATE'], fedfundsDF['FEDFUNDS_PERC'], 
             marker='o', linestyle='-', color="red", label="fedfunds")
print(fedfundsDF.dtypes)
print(revenue_df.dtypes)
colors = ['purple', 'blue', 'orange', 'green', 'pink']
for i, ticker in enumerate(top_5_tickers):
    company_data = revenue_df[revenue_df['ticker'] == ticker]
    company_data['value_percentage'] = (company_data['value'] / company_data['value'].max()) * 100
    plt.plot(company_data['interval_date'], company_data['value_percentage'], 
             marker='o', linestyle='-', color=colors[i], label=ticker)
plt.title('Revenue Over Time for Multiple Companies')
plt.xlabel('Date')
plt.ylabel('Value%')
plt.grid(True)
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('analysis/Revenue Multiple Companies.png')
plt.show()
########

# Apply the function to both DataFrames
#startDate = int(input("Enter the start date:"))
#endDate = int(input("Enter the end date:"))
startDate = pd.to_datetime('2005-01')
endDate = pd.to_datetime('2024-01')
farmingCompanies = filter_and_drop(farmingCompanies, 'interval_date', startDate, endDate)

#interval = int(input("Enter (months) interval:"))
interval = 0
fedfundsDF = filter_and_drop(fedfundsDF, 'DATE', startDate-pd.DateOffset(months=interval), endDate+pd.DateOffset(months=interval))
wheatDF = filter_and_drop(wheatDF, 'Date', startDate-pd.DateOffset(months=interval), endDate+pd.DateOffset(months=interval))
base_date = farmingCompanies.iloc[0, 0].date()
        
valid_months = [
    (1 + base_date.month - interval) % 12,
    (4 + base_date.month - interval) % 12,
    (7 + base_date.month - interval) % 12,
    (10 + base_date.month - interval) % 12 
]
# Extracting names of companies from file
unique_names = set()
farmingCompanies['interval_date'] = pd.to_datetime(farmingCompanies['interval_date'], infer_datetime_format=True)
for name in farmingCompanies['ticker']:
    unique_names.add(name)
    
fedfundsDF['DATE'] = pd.to_datetime(fedfundsDF['DATE'])
fedfundsDF = fedfundsDF[fedfundsDF['DATE'].dt.month.isin(valid_months)]
wheatDF = wheatDF[wheatDF['Date'].dt.month.isin(valid_months)]
for name in unique_names:
    currentCompany=farmingCompanies[farmingCompanies['ticker'] == name]
    plt.figure(figsize=(8, 6))  # Adjust the figure size for better readability
    # Getting needed elements
    add_to_cor(corDF, currentCompany[currentCompany['item_name']== 'freeCashFlow']['value'] , 'freeCashFlow')
    add_to_cor(corDF, currentCompany[currentCompany['item_name']== 'netIncome']['value'] , 'netIncome')
    add_to_cor(corDF, currentCompany[currentCompany['item_name']== 'totalRevenue']['value'] , 'totalRevenue')
    add_to_cor(corDF, fedfundsDF['FEDFUNDS'], 'FEDFUNDS')
    add_to_cor(corDF, wheatDF['Volume'], 'Volume')
    #creating correlation matrix + displaying/saving it
    corr_matrix = corDF.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')
    plt.legend(title='First financial report: ' + str(currentCompany.iloc[0,0].date()) + ' Fedfunds start: ' + str(fedfundsDF.iloc[0,0].date()) + ' Grain prices:' + str(wheatDF.iloc[0,0].date()), bbox_to_anchor=(-0.04, 1.15), loc='upper left')
    plt.title(name)
    plt.savefig('analysis/' + name + '.png', dpi=300, bbox_inches='tight') 
    plt.show()

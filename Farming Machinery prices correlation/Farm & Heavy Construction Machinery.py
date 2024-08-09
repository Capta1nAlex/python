import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns

# Read a CSV file into a DataFrame
farmingCompanies = pd.read_csv('Farm.csv') #Data can be found in public sources
fedfundsDF = pd.read_csv('FEDFUNDS.csv')
wheatDF = pd.read_csv('Wheat.csv')
farmingCompanies=farmingCompanies.dropna()
fedfundsDF=fedfundsDF.dropna()
wheatDF=wheatDF.dropna()
def filter_and_drop(df, column_to_filter):
    # Convert the date column to datetime
    df[column_to_filter] = pd.to_datetime(df[column_to_filter])
    # Identify rows where the date is older than 2004 or older than 2023
    rows_to_drop = df[(df[column_to_filter].dt.year < 2005) | (df[column_to_filter].dt.year > 2023)].index
    df.drop(index=rows_to_drop, inplace=True)
    # Check if any value in the date column for each column is older than 2004 or older than 2023
    columns_to_keep = [column_to_filter] + [col for col in df.columns if col != column_to_filter and not any((df[column_to_filter].dt.year < 2004) | (df[column_to_filter].dt.year > 2023))]
    # Create a new DataFrame with only the columns to keep
    return df[columns_to_keep]


# Apply the function to both DataFrames
farmingCompanies = filter_and_drop(farmingCompanies, 'interval_date')
fedfundsDF = filter_and_drop(fedfundsDF, 'DATE')
wheatDF = filter_and_drop(wheatDF, 'Date')
valid_months = [1, 4, 7, 10]

# Extracting names of companies from file
unique_names = set()
farmingCompanies['interval_date'] = pd.to_datetime(farmingCompanies['interval_date'], infer_datetime_format=True)
for name in farmingCompanies['ticker']:
    unique_names.add(name)

for name in unique_names:
    fedfundsDF['DATE'] = pd.to_datetime(fedfundsDF['DATE'])
    fedfundsDF = fedfundsDF[fedfundsDF['DATE'].dt.month.isin(valid_months)]
    wheatDF = wheatDF[wheatDF['Date'].dt.month.isin(valid_months)]
    currentCompany=farmingCompanies[farmingCompanies['ticker'] == name]
    plt.figure(figsize=(8, 6))  # Adjust the figure size for better readability
    
    corDF=currentCompany[currentCompany['item_name'] == "freeCashFlow"]
    corDF = corDF.reset_index(drop=True)
    # Getting needed elements
    tempColumn=currentCompany[currentCompany['item_name'] == "freeCashFlow"]
    tempColumn = tempColumn.reset_index(drop=True)
    corDF['freeCashFlow']=tempColumn['value']

    tempColumn=currentCompany[currentCompany['item_name']== 'netIncome']
    tempColumn = tempColumn.reset_index(drop=True)
    corDF['netIncome']=tempColumn['value']

    tempColumn=currentCompany[currentCompany['item_name']== 'totalRevenue']
    tempColumn = tempColumn.reset_index(drop=True)
    corDF['totalRevenue']=tempColumn['value']
    
    tempColumn=fedfundsDF['FEDFUNDS']
    tempColumn = tempColumn.reset_index(drop=True)
    corDF['FEDFUNDS']=tempColumn
    tempColumn=wheatDF['Volume']
    tempColumn = tempColumn.reset_index(drop=True)
    corDF['Volume']=tempColumn
    #creating correlation matrix + displaying/saving it
    ratio = corDF[['freeCashFlow', 'netIncome', 'totalRevenue', 'FEDFUNDS', 'Volume']]
    corr_matrix = ratio.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')
    plt.legend(title='Fed funds start: ' + str(currentCompany.iloc[0,0].date()) + ' First financial report: ' + str(farmingCompanies.iloc[0,0].date()) + ' Grain prices:' + str(wheatDF.iloc[0,0].date()), bbox_to_anchor=(-0.04, 1.15), loc='upper left')
    plt.title(name)
    plt.savefig('analysis/' + name + '.png', dpi=300, bbox_inches='tight') 
    plt.show()

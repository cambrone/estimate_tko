import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


def summary_stats(df, col):
    print('pct missing: ', df[col].isnull().sum()/df.shape[0])
    print('pct zeros: ', df[df[col]==0].shape[0]/df.shape[0])
    print('min: ', df[col].min())
    print('mean: ', df[col].mean())
    print('max: ', df[col].max())
    

def change_over_time(df, col, time_col):
    temp_val = pd.DataFrame(df.groupby(time_col, dropna=False)['val'].sum()).reset_index()
    temp_mean = pd.DataFrame(df.groupby(time_col)[col].mean()).reset_index()
    temp_mean = temp_mean.rename(columns={col:'mean'})

    temp_q25 = pd.DataFrame(df.groupby(time_col)[col].quantile(0.25)).reset_index()
    temp_q25 = temp_q25.rename(columns={col:'q25'})
    
    temp_q50 = pd.DataFrame(df.groupby(time_col)[col].quantile(0.5)).reset_index()
    temp_q50 = temp_q50.rename(columns={col:'q50'})

    temp_q75 = pd.DataFrame(df.groupby(time_col)[col].quantile(0.75)).reset_index()
    temp_q75 = temp_q75.rename(columns={col:'q75'})

    temp = temp_val.merge(temp_mean, on =time_col, how ='left')
    temp = temp.merge(temp_q25, on = time_col, how ='left')
    temp = temp.merge(temp_q50, on = time_col, how ='left')
    temp = temp.merge(temp_q75, on = time_col, how ='left')

    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()

    temp['val'].plot(kind='bar', ax=ax2, alpha = 0.15)
    temp['mean'].plot(kind='line', marker='d', ax=ax1, color='red')
    temp['q25'].plot(kind='line', marker='.', ax=ax1, color='blue')
    temp['q50'].plot(kind='line', marker='.', ax=ax1, color='green')
    temp['q75'].plot(kind='line', marker='.', ax=ax1, color='blue')

    plt.title(col+' over time')
    ax1.set_xlabel(time_col)
    ax1.set_ylabel(col)
    ax2.set_ylabel("fights")
    ax1.set_xticklabels(temp[time_col],  rotation=45)

    legend = ax1.legend(loc='upper right')

    plt.show()
    
def change_over_time_by_target(df, target, col, time_col):
    #variable changes overtime by early decsion 
    temp_val = pd.DataFrame(df.groupby([time_col,target], dropna=False)['val'].sum()).reset_index()
    temp_mean = pd.DataFrame(df.groupby([time_col,target])[col].mean()).reset_index()
    temp_mean = temp_mean.rename(columns={col:'mean'})

    temp = temp_val.merge(temp_mean, on = [time_col,target], how ='left')

    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()

    temp[temp[target]==0]['val'].plot(kind='bar', ax=ax2, alpha = 0.15, color='black')
    temp[temp[target]==1]['val'].plot(kind='bar', ax=ax2, alpha = 0.15, color='tab:red')

    temp[temp[target]==0]['mean'].plot(kind='line', marker='d', ax=ax1, color='black')
    temp[temp[target]==1]['mean'].plot(kind='line', marker='d', ax=ax1, color='tab:red')

    plt.title(col+' over time')
    ax1.set_xlabel(time_col)
    ax1.set_ylabel(col)
    ax2.set_ylabel("fights")
    ax1.set_xticklabels(temp[temp[target]==1][time_col],  rotation=45)

    legend = ax1.legend(["decision", "early_stop"] ,loc='upper right')

    plt.show()
    
def exploratory_lift_chart(df, target,col, nbins):

        ser, bins = pd.qcut(df[col], nbins, retbins=True, duplicates='drop')
        df[col+'_group'] = pd.cut(df[col], bins=bins)

        temp_sum = pd.DataFrame(df.groupby([col+'_group'], dropna=False)['val'].sum()).reset_index()
        temp_q25 = pd.DataFrame(df.groupby([col+'_group'], dropna=False)[target].quantile(0.25)).reset_index()
        temp_q25 = temp_q25.rename(columns={'early_stop':'q25'})

        temp_mean = pd.DataFrame(df.groupby([col+'_group'], dropna=False)[target].mean()).reset_index()
        temp_mean = temp_mean.rename(columns={'early_stop':'mean'})

        temp_q50 = pd.DataFrame(df.groupby([col+'_group'], dropna=False)[target].quantile(0.50)).reset_index()
        temp_q50 = temp_q50.rename(columns={'early_stop':'q50'})

        temp_q75 = pd.DataFrame(df.groupby([col+'_group'], dropna=False)[target].quantile(0.75)).reset_index()
        temp_q75 = temp_q75.rename(columns={'early_stop':'q75'})

        temp = temp_sum.merge(temp_q25, on = col+'_group', how ='left')
        temp = temp.merge(temp_mean, on = col+'_group', how ='left')
        temp = temp.merge(temp_q50, on = col+'_group', how ='left')
        temp = temp.merge(temp_q75, on = col+'_group', how ='left')

        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax2 = ax1.twinx()

        temp['val'].plot(kind='bar', ax=ax2, alpha = 0.15)
        temp['mean'].plot(kind='line', marker='d', ax=ax1, color='red')
        temp['q25'].plot(kind='line', marker='.', ax=ax1, color='blue')
        temp['q50'].plot(kind='line', marker='.', ax=ax1, color='green')
        temp['q75'].plot(kind='line', marker='.', ax=ax1, color='blue')

        plt.title(col+'_group with debut')
        ax1.set_xlabel(col+'_group')
        ax1.set_ylabel("mean "+target)
        ax2.set_ylabel("fights")
        ax1.set_xticklabels(temp[col+'_group'],  rotation=45)

        legend = ax1.legend(loc='upper right')

        plt.show()
        
def change_over_time_cat(df, col, time_col):
    
    temp = pd.DataFrame(df.groupby([time_col, col])['val'].sum()/df.groupby([time_col])['val'].sum()).reset_index()
    temp = temp.pivot(index=time_col, columns=col, values='val').reset_index()

    temp_columns = list(temp.columns)
    temp_columns.remove(time_col)
    
    fig, ax1 = plt.subplots(figsize=(10, 10))

    for col in temp_columns:
        temp[col].plot(kind='line', marker='d', ax=ax1)
    
    plt.title('proportions over time')
    ax1.set_xlabel(time_col)
    ax1.set_ylabel('pct '+col)
    ax1.set_xticklabels(temp[time_col],  rotation=45)

    legend = ax1.legend(loc='upper right')

    plt.show()  
    
def change_over_time_by_target_cat(df, target, col, time_col):
    
    temp = pd.DataFrame(df.groupby([time_col,col])[target].mean()).reset_index()
    temp = temp.pivot(index=time_col, columns=col, values=target).reset_index()

    temp_columns = list(temp.columns)
    temp_columns.remove(time_col)
    
    fig, ax1 = plt.subplots(figsize=(10, 10))

    for col in temp_columns:
        temp[col].plot(kind='line', marker='d', ax=ax1)
    
    plt.title(target+' over time')
    ax1.set_xlabel(time_col)
    ax1.set_ylabel('pct '+col)
    ax1.set_xticklabels(temp[time_col],  rotation=45)

    legend = ax1.legend(loc='upper right')

    plt.show()  
    
def group_mean_cat(df, target, col):
    temp = pd.DataFrame(df.groupby([col])[target].mean()).reset_index()
    temp = temp.sort_values(target, ascending=True)
    fig, ax1 = plt.subplots(figsize=(10, 10))

    temp[target].plot(kind='bar', ax=ax1, color=plt.cm.Paired(np.arange(len(temp))))
    
    plt.title(target)
    ax1.set_xlabel(col)
    ax1.set_ylabel('pct '+target)
    ax1.set_xticklabels(temp[col],  rotation=45)

    plt.show()  
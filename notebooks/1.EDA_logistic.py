#%%
import pandas as pd
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt

#%%
pd.set_option('display.max_columns', None)

from config import clean_data_path
from utils.data_exploration import summary_stats,change_over_time,change_over_time_by_target,exploratory_lift_chart, change_over_time_cat,change_over_time_by_target_cat,group_mean_cat

final_df = pd.read_csv(clean_data_path+'final_fight_df_binary.csv')
final_df['val'] = 1


final_df['event_date'] = pd.to_datetime(final_df['event_date'])
final_df['year'] = pd.to_datetime(final_df['event_date']).dt.year
final_df['month_year'] = final_df['event_date'].dt.to_period('Q')
final_df.groupby(['year'])['early_stop'].mean()

#%%
print('early stop by year: ',final_df[final_df['year']>=2013].groupby(['year'])['early_stop'].mean())
final_df = final_df[final_df['year']>=2013].copy()

#%%
var_cols =  sorted(final_df.columns[(final_df.columns.str.endswith('_A_to_B_diff')) | 
                                    (final_df.columns.str.endswith('_abs_A_minus_B')) | 
                                    (final_df.columns.str.endswith('_A_plus_B')) |
                                    (final_df.columns.str.endswith('_A_plus_B_per_total_fights'))])



#%%
no_debut = final_df[(final_df['debut_A']==0) & (final_df['debut_B']==0)].copy()

for col in var_cols:
    print('*********'+col+'*********')
    if len(final_df[col].unique())<5:
        continue
    else:
        summary_stats(df= final_df, col=col)

        #variable changes overtime
        change_over_time(df= final_df, col=col, time_col= 'month_year')

        #variable changes overtime by early decsion 
        change_over_time_by_target(df = final_df, target = 'early_stop', col = col, time_col='month_year')

        #early decision over buckets of variable
        exploratory_lift_chart(df = final_df, target = 'early_stop', col = col, nbins=5)

        #early decision over buckets of variable no debut
        exploratory_lift_chart(df = no_debut, target = 'early_stop', col = col, nbins=5)

#%%
#categorical variables
cat_vars = ['weight_class', 'gender']

for var in cat_vars:
    print('************'+var+'*************')

    print(final_df.groupby(var, dropna=False)['val'].sum()/final_df['val'].sum())
    
    # pct of each group over time
    change_over_time_cat(df= final_df, col=var, time_col='year')
    
    # for each group, mean early stop over time
    change_over_time_by_target_cat(df= final_df, target='early_stop', col=var, time_col='year')

    # average early stop 
    group_mean_cat(df=final_df, target='early_stop', col=var)

#%%
#correlation among variables
corr_matrix = final_df[['early_stop'] + var_cols].corr()
corr_matrix['early_stop_abs'] =np.abs(corr_matrix['early_stop'])
temp_corr = corr_matrix[['early_stop', 'early_stop_abs']]
temp_corr = temp_corr.sort_values('early_stop_abs', ascending=False)

#%%
#%%
import pandas as pd
import numpy as np 
from datetime import datetime
import random 

np.random.seed(222)

#%%
pd.set_option('display.max_columns', None)

from config import clean_data_path
from utils.data_exploration import summary_stats,change_over_time,change_over_time_by_target,exploratory_lift_chart, change_over_time_cat,change_over_time_by_target_cat,group_mean_cat

logistic_df = pd.read_csv(clean_data_path+'final_fight_df_binary.csv')
logistic_df['val'] = 1


logistic_df['event_date'] = pd.to_datetime(logistic_df['event_date'])
logistic_df['year'] = pd.to_datetime(logistic_df['event_date']).dt.year
logistic_df['month_year'] = logistic_df['event_date'].dt.to_period('Q')
logistic_df.groupby(['year'])['early_stop'].mean()

#%%
#average the events used in logistic model
mean_df = pd.DataFrame(logistic_df.groupby(['event_id',
                                            'event_name',
                                            'event_date',
                                            'partition'])[['unanimous_decision_per_round_abs_A_minus_B',
                                                           'early_stop_A_plus_B_per_total_fights',
                                                           'fighter_reach_cm_A_plus_B',
                                                           'switch_A_plus_B',
                                                           'decision_A_plus_B']].mean()).reset_index()
mean_df = mean_df.add_suffix('_mean')
mean_df = mean_df.rename(columns={'event_id_mean': 'event_id',
                                  'event_name_mean': 'event_name',
                                  'event_date_mean': 'event_date',
                                  'partition_mean':'partition'})

sum_df = pd.DataFrame(logistic_df.groupby(['event_id',
                                           'event_name',
                                           'event_date',
                                           'partition'])[['unanimous_decision_per_round_abs_A_minus_B',
                                                          'early_stop_A_plus_B_per_total_fights',
                                                          'fighter_reach_cm_A_plus_B',
                                                          'switch_A_plus_B',
                                                          'decision_A_plus_B']].sum()).reset_index()                    
sum_df = sum_df.add_suffix('_sum')
sum_df = sum_df.rename(columns={'event_id_sum': 'event_id',
                                  'event_name_sum': 'event_name',
                                  'event_date_sum': 'event_date',
                                  'partition_sum':'partition'})



#sum the early stops
agg_target = pd.DataFrame(logistic_df.groupby(['event_id',
                                           'event_name',
                                           'event_date',
                                           'partition'])['early_stop'].sum()).reset_index()      

poisson_df = mean_df.merge(sum_df, on = ['event_id',
                                         'event_name',
                                         'event_date',
                                         'partition'],
                           how='left')              

poisson_df = poisson_df.merge(agg_target, on = ['event_id',
                                                'event_name',
                                                'event_date',
                                                'partition'],
                              how='left') 

#%%     
########################################################################
# Attache the boot strap confidence intervals from Binary
########################################################################
ci_logistic = pd.read_csv(clean_data_path+'ci_logistic.csv')
ci_logistic['event_date'] = pd.to_datetime(ci_logistic['event_date'])
ci_logistic = ci_logistic.drop(['early_stop'], axis=1)	
ci_logistic = ci_logistic.rename(columns={'ci_low':'ci_low_binary',
                                          'ci_high':'ci_high_binary'})

#%%
print(poisson_df.shape[0])
poisson_df = poisson_df.merge(ci_logistic, on = ['event_id',
                                                'event_name',
                                                'event_date',
                                                'partition'], how='inner')
print(poisson_df.shape[0])

#%%

poisson_df.to_csv(clean_data_path+'final_fight_df_possion.csv', index=False)
#%%

#%%
import pandas as pd
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.interpolate import make_interp_spline, BSpline
from scipy.special import expit
import random 

np.random.seed(222)
    
pd.set_option('display.max_columns', None)

from config import clean_data_path, model_path
from utils.model_eval import score_w_model, partial_resid_plot, calculate_nested_f_statistic, lift_chart, var_importance_plot, factor_plot, perf_metrics_poisson
from utils.data_exploration import summary_stats,change_over_time,change_over_time_by_target,exploratory_lift_chart, change_over_time_cat,change_over_time_by_target_cat,group_mean_cat


#%%
################################
# LOAD DATA AND DEFINE TARGET
################################
final_df = pd.read_csv(clean_data_path+'final_fight_df_possion.csv')

# define target
target = 'early_stop'

# data quality check
final_df['event_date'] = pd.to_datetime(final_df['event_date'])
final_df['year'] = pd.to_datetime(final_df['event_date']).dt.year
final_df['month_year'] = final_df['event_date'].dt.to_period('Q')
final_df.groupby(['year'])['early_stop'].mean()

final_df['val'] = 1
#%%
################################ 
# PARTITION DATA 
################################
# define datasets to be standardized 
train_df = final_df[final_df['partition']=='T'].copy()
val_df = final_df[final_df['partition']=='V'].copy()
train_val_df = final_df[final_df['partition']!='H'].copy() 

#%%
##########################
# Explore Target
########################## 
train_df[target].hist(bins=10)


#%%
########################
# Explore data 
########################
predictor_list = [
    'unanimous_decision_per_round_abs_A_minus_B_mean',
    'early_stop_A_plus_B_per_total_fights_mean',
    'fighter_reach_cm_A_plus_B_mean',
    'switch_A_plus_B_mean',
    'decision_A_plus_B_mean',
    ]

for col in predictor_list:
    print('*********'+col+'*********')
    if len(final_df[col].unique())<5:
        continue
    else:
        ser, bins = pd.qcut(final_df[final_df['partition']=='T'][col], 10, retbins=True, duplicates='drop')
        final_df[col+'_grp'] = pd.cut(final_df[col], bins=bins)
        final_df[col+'_grp'] = final_df[col+'_grp'].astype(str)
        
        temp_val = pd.DataFrame(final_df.groupby(col+'_grp')['val'].sum()).reset_index()

        temp_mean = pd.DataFrame(final_df.groupby(col+'_grp')[target].mean()).reset_index()
        temp_mean = temp_mean.rename(columns={target:target+'_mean'})
        
        temp_var = pd.DataFrame(final_df.groupby(col+'_grp')[target].var()).reset_index()
        temp_var = temp_var.rename(columns={target:target+'_var'})
        
        temp = temp_val.merge(temp_mean, on = col+'_grp', how='left')
        temp = temp.merge(temp_var, on = col+'_grp', how='left')

        print('****mean - variance *****')
        print(temp)
        
        summary_stats(df = final_df, col = col)
        
        #early decision over buckets of variable
        exploratory_lift_chart(df = final_df, target = 'early_stop', col = col, nbins = 5)

#%%
model2 = sm.GLM(train_df[target], sm.add_constant(train_df[predictor_list]), family=sm.families.Poisson()).fit()

print(model2.summary())

#%%
train_val_df = score_w_model(df=train_val_df,
                                target= target,
                                predictor_list = predictor_list,
                                model_version = model2,
                                model_name = 'model2')

train_val_df.groupby('partition').apply(perf_metrics_poisson, target, 'pred_model2', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=train_val_df, 
           obs=target, 
           pred='pred_model2', 
           partition='partition',
           count='val',
           nbin=10)


#%%
# variable importance    
var_importance_plot(model_version = model2)

#%%
# functional form with partial dependence plots
partial_resid_plot(df = train_val_df, 
                   predictor_list = predictor_list, 
                   work_resid = 'resid_working_model2',
                   model_version = model2, 
                   lowess_frac=0.3)

#%%       
factor_plot(df = train_val_df,
            obs = target,
            pred = 'pred_model2',
            count = 'val', 
            partition ='partition', 
            model_version = model2,
            predictor_list = predictor_list,
            factor=False)  

#%%
##################
# SCORE holdout 
##################
final_df = score_w_model(df=final_df,
                         target= target,
                         predictor_list = predictor_list,
                         model_version = model2,
                         model_name = 'model2')

final_df.groupby('partition').apply(perf_metrics_poisson, target, 'pred_model2', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=final_df, 
           obs=target, 
           pred='pred_model2', 
           partition='partition',
           count='val',
           nbin=10)


temp = pd.DataFrame(final_df.groupby(['month_year'])['early_stop'].mean()).reset_index()

temp.plot(x='month_year', y='early_stop',kind='line', marker='d')


#%%
###################################
# Add prediction confidence intervals
###################################
boot_final_df = final_df[['event_id','event_name','event_date','partition',target]].copy()

train_sample = final_df[final_df['partition']=='T'].copy()
sample_size = train_sample.shape[0]
n_boot = 1000

boot_coeff_agg = pd.DataFrame()
boot_event_early_stop_agg = pd.DataFrame()

for i in range(0,n_boot):
    boot_sample = train_sample.iloc[np.random.randint(sample_size, size=sample_size)]
    boot_model = sm.GLM(boot_sample[target], sm.add_constant(boot_sample[predictor_list]), family=sm.families.Poisson()).fit()

    # bootstrapped coeff
    boot_coeff = pd.DataFrame(boot_model.params)
    boot_coeff = boot_coeff.rename(columns={ 0:'boot_sample_'+str(i)})
    
    boot_coeff_agg = pd.concat([boot_coeff_agg, boot_coeff], axis=1)
    
    #bootstrapped pred
    boot_final_df['boot_pred'] = boot_model.predict((sm.add_constant(final_df[predictor_list])))
        
    boot_final_df['resid'] = boot_final_df[target] - boot_final_df['boot_pred']
    residuals = np.random.choice(boot_final_df['resid'], boot_final_df.shape[0])
    # residuals = np.random.choice(boot_final_df[boot_final_df['partition']=='T']['resid'], boot_final_df.shape[0])

    boot_final_df['boot_pred'] = boot_final_df['boot_pred'] + residuals
    boot_final_df = boot_final_df.drop(['resid'], axis=1)
    
    boot_final_df = boot_final_df.rename(columns={'boot_pred':'boot_pred_'+str(i)})
        
    if i % 1000 == 0:
        print('loop '+ str(i))
    
#%%
boot_cols = list(boot_final_df.columns[boot_final_df.columns.str.startswith('boot_pred_')])

boot_final_df['ci_low_poisson'] = boot_final_df[boot_cols].quantile(0.025, axis=1)
boot_final_df['ci_high_poisson'] =  boot_final_df[boot_cols].quantile(0.975, axis=1)

boot_final_df['within_range'] = np.nan
boot_final_df['within_range'] = np.where((boot_final_df['ci_low_poisson']<=boot_final_df[target]) &
                                         (boot_final_df[target]<=boot_final_df['ci_high_poisson']),1,0)

print('obs wihtin range: ', boot_final_df['within_range'].mean())

boot_final_df = boot_final_df[['event_id',
                               'event_name',
                               'event_date',
                               'partition',
                               'ci_low_poisson',
                               'ci_high_poisson']].copy()

print(final_df.shape[0])
final_df = final_df.merge(boot_final_df, on=['event_id',
                               'event_name',
                               'event_date',
                               'partition'], how='left')
print(final_df.shape[0])
print('missing rate: ', final_df[['ci_low_poisson','ci_high_poisson']].isnull().sum())
#%%

final_df['ci_width_binary'] = final_df['ci_high_binary'] - final_df['ci_low_binary']
final_df['ci_width_poisson'] = final_df['ci_high_poisson'] - final_df['ci_low_poisson']


temp_binary = pd.DataFrame(final_df.groupby('partition')[['ci_width_binary']].describe())
temp_binary = temp_binary.add_suffix('_binary')

temp_poisson = pd.DataFrame(final_df.groupby('partition')[['ci_width_poisson']].describe())
temp_poisson = temp_poisson.add_suffix('_poisson')

#%%
temp = temp_binary.merge(temp_poisson, on='partition', how='left')
temp.columns = temp.columns.droplevel()
temp = temp.reset_index()


#using binary model and calculating the 95% prediciton interval leads to narrower CI than poisson model
temp = temp[['partition', 'mean_binary', 'std_binary',
             'mean_poisson', 'std_poisson']]

print(temp)
# %%

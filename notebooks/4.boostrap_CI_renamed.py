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

np.random.seed(111)
    
pd.set_option('display.max_columns', None)

from config import clean_data_path, model_path
from utils.model_eval import score_w_model, partial_resid_plot, calculate_nested_f_statistic, lift_chart, var_importance_plot, factor_plot, perf_metrics
from utils.data_exploration import summary_stats,change_over_time,change_over_time_by_target,exploratory_lift_chart, change_over_time_cat,change_over_time_by_target_cat,group_mean_cat



#%%
################################
# LOAD DATA AND DEFINE TARGET
################################
final_df = pd.read_csv(clean_data_path+'final_fight_df_binary.csv')

# define target
target = 'early_stop'

# data quality check
final_df['event_date'] = pd.to_datetime(final_df['event_date'])
final_df['year'] = pd.to_datetime(final_df['event_date']).dt.year
final_df['month_year'] = final_df['event_date'].dt.to_period('Q')
final_df.groupby(['year'])['early_stop'].mean()

final_df['val'] = 1

#define predictors
predictor_list = [
    'unanimous_decision_per_round_abs_A_minus_B',
    'early_stop_A_plus_B_per_total_fights',
    'fighter_reach_cm_A_plus_B',
    'switch_A_plus_B',
    'decision_A_plus_B',
    ]

#load model and score dataset
with open(model_path+"logistic_model.pkl", "rb") as f:
    model1 = pickle.load(f)
    

final_df = score_w_model(df=final_df,
                         target= target,
                         predictor_list = predictor_list,
                         model_version = model1,
                         model_name = 'model1')


#%%
#######################################################################
# (aggregate each bootstrap sample to event level and calc residual)
#######################################################################
train_sample = final_df[final_df['partition']=='T'].copy()
sample_size = train_sample.shape[0]
n_boot = 1000

boot_coeff_agg = pd.DataFrame()
boot_event_early_stop_agg = pd.DataFrame()

for i in range(0,n_boot):
    boot_sample = train_sample.iloc[np.random.randint(sample_size, size=sample_size)]
    boot_model = sm.GLM(boot_sample[target], sm.add_constant(boot_sample[predictor_list]), family=sm.families.Binomial()).fit()

    # bootstrapped coeff
    boot_coeff = pd.DataFrame(boot_model.params)
    boot_coeff = boot_coeff.rename(columns={ 0:'boot_sample_'+str(i)})
    
    boot_coeff_agg = pd.concat([boot_coeff_agg,boot_coeff], axis=1)
    
    #bootstrapped pred
    final_df['boot_pred'] = boot_model.predict((sm.add_constant(final_df[predictor_list])))

    boot_event_early_stops = pd.DataFrame(final_df.groupby(['event_id','partition'])[[target,'boot_pred']].sum())
        
    boot_event_early_stops['resid'] = boot_event_early_stops[target] - boot_event_early_stops['boot_pred']
    boot_event_early_stops = boot_event_early_stops.reset_index(level=['partition'])
    
    residuals = np.random.choice(boot_event_early_stops[boot_event_early_stops['partition']=='T']['resid'], boot_event_early_stops.shape[0])

    boot_event_early_stops['boot_pred'] = boot_event_early_stops['boot_pred'] + residuals
    boot_event_early_stops = boot_event_early_stops.drop([target, 'resid','partition'], axis=1)
    
    boot_event_early_stops = boot_event_early_stops.rename(columns={'boot_pred':'boot_pred_'+str(i)})
    boot_event_early_stop_agg =  pd.concat([boot_event_early_stop_agg, boot_event_early_stops], axis=1)
    
    if i % 1000 == 0:
        print('loop '+ str(i))

#%%
################################################################################
# Evaluate expected number of early stops 
################################################################################
final_df = final_df.sort_values('event_date', ascending=True)

final_df['random_uniform'] = np.random.uniform(low=0,
                                              high=1,
                                              size=final_df.shape[0])

final_df['random_choice'] = np.random.randint(low=0,
                                              high=2,
                                              size=final_df.shape[0])

final_df['overall_avg'] = final_df[final_df['partition']=='T'][target].mean()

#%%
#aggregate to event level
compare_cols = ['pred_model1', 'random_uniform','random_choice', 'overall_avg']

event_early_stops = pd.DataFrame(final_df.groupby(['event_id'
                                                   ,'event_name','event_date',
                                                   'partition'])[['early_stop',
                                                                  *compare_cols]].sum()).reset_index()

event_early_stops['val'] = 1

#plot errors
error_cols = []
for var in compare_cols:
    event_early_stops['abs_error_'+var] = np.abs(event_early_stops['early_stop'] - event_early_stops[var])
    error_cols = error_cols + ['abs_error_'+var]

error_results = pd.DataFrame(event_early_stops.groupby('partition')[error_cols].mean()).reset_index()

error_results['order'] = np.nan
error_results['order'] = np.where(error_results['partition']=='T', 1, error_results['order'])
error_results['order'] = np.where(error_results['partition']=='V', 2, error_results['order'])
error_results['order'] = np.where(error_results['partition']=='H', 3, error_results['order'])

error_results = error_results.sort_values(['order'], ascending=True)
error_results = error_results.drop('order', axis=1)

error_results = error_results.set_index(list(error_results)[0])

ax = error_results.plot(kind='bar', figsize=(20, 8), rot=0, ylabel='abs error', title="absolute error comparisons")

for c in ax.containers:
    ax.bar_label(c, fmt='%.3f', label_type='edge')
    
#%%
# lift chart of events
temp_full = event_early_stops.copy()
temp_full = temp_full.sort_values('early_stop', ascending=True)

ser, bins = pd.qcut(temp_full[temp_full['partition']=='T']['early_stop'], 5, retbins=True, duplicates='drop')
temp_full['early_stop_grp'] = pd.cut(temp_full['early_stop'], bins=bins, include_lowest=True)
temp_full['early_stop_grp'] = temp_full['early_stop_grp'].astype(str)

for part in ['T','V','H']:
    temp = temp_full[(temp_full['partition']==part)].copy()

    cnt = pd.DataFrame(temp.groupby(['early_stop_grp'])['val'].sum()).reset_index()
    mean = pd.DataFrame(temp.groupby(['early_stop_grp'])[['early_stop',*compare_cols]].mean()).reset_index()
    temp = cnt.merge(mean, on = ['early_stop_grp'], how='left')
        
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()
        
    temp['val'].plot(kind='bar', ax=ax2, alpha = 0.15)
    temp['early_stop'].plot(kind='line', marker='d', ax=ax1, color='tab:red')
    temp['pred_model1'].plot(kind='line', marker='.', ax=ax1, color='tab:blue')
    temp['random_uniform'].plot(kind='line', marker='.', ax=ax1, color='tab:green')
    temp['random_choice'].plot(kind='line', marker='.', ax=ax1, color='tab:purple')
    temp['overall_avg'].plot(kind='line', marker='.', ax=ax1, color='tab:brown')

    plt.title(part)
    ax1.set_xlabel('early_stop_grp')
    ax1.set_ylabel('early_stop')
    ax2.set_ylabel('val')
    ax1.set_xticklabels(temp['early_stop_grp'],  rotation=45)
    
    ax1.legend(loc="upper right")

    plt.show()

#%%
#######################################
# compare classical v bootstrapped CI
#######################################
parameters = pd.DataFrame(model1.params).reset_index()
parameters = parameters.rename(columns={'index':'predictor',
                                        0:'coeff'})

classic_ci = pd.DataFrame(model1.conf_int()).reset_index()
classic_ci = classic_ci.rename(columns={'index':'predictor',
                                        0:'ci_low_classic',
                                        1:'ci_high_classic'})


boot_ci_cols = list(boot_coeff_agg.columns[boot_coeff_agg.columns.str.startswith('boot_sample_')])
boot_coeff_agg['ci_low_boot'] = boot_coeff_agg[boot_ci_cols].quantile(0.025, axis=1)
boot_coeff_agg['ci_high_boot'] = boot_coeff_agg[boot_ci_cols].quantile(0.975, axis=1)
boot_coeff_agg = boot_coeff_agg.reset_index()
boot_coeff_agg = boot_coeff_agg.rename(columns={'index':'predictor'})

conf_int = parameters.merge(classic_ci, on='predictor', how='left')
conf_int = conf_int.merge(boot_coeff_agg[['predictor', 'ci_low_boot', 'ci_high_boot']], 
                          on='predictor',
                          how='left')

conf_int = conf_int[conf_int['predictor']!='const']

#%%
import matplotlib.patches as mpatches

fig, ax1 = plt.subplots(figsize=(10, 10))

ax1.plot((0,2,4,6,8),conf_int['coeff'], marker='o' , color='black', linestyle='None')
ax1.plot((0,2,4,6,8),conf_int['ci_low_classic'], marker='_', markersize=15 ,color='blue', linestyle='None', mew=3)
ax1.plot((0,2,4,6,8),conf_int['ci_high_classic'],marker='_', markersize=15,color='blue', linestyle='None', mew=3)

ax1.plot((0.5,2.5,4.5,6.5,8.5), conf_int['coeff'], marker='o' , color='black', linestyle='None')
ax1.plot((0.5,2.5,4.5,6.5,8.5),conf_int['ci_low_boot'], marker='_', markersize=15 ,color='green', linestyle='None', mew=3)
ax1.plot((0.5,2.5,4.5,6.5,8.5),conf_int['ci_high_boot'],marker='_', markersize=15,color='green', linestyle='None', mew=3)

ax1.set_xticks((0,2,4,6,8))
ax1.set_xticklabels(conf_int['predictor'],  rotation=90)

blue_patch = mpatches.Patch(color='blue', label='classic CI')
green_patch = mpatches.Patch(color='green', label='boot CI')
plt.legend(handles=[blue_patch, green_patch],loc='lower right')
plt.show()


# %%
boot_cols = list(boot_event_early_stop_agg.columns[boot_event_early_stop_agg.columns.str.startswith('boot_pred_')])

boot_event_early_stop_agg['ci_low'] = boot_event_early_stop_agg[boot_cols].quantile(0.025, axis=1)
boot_event_early_stop_agg['ci_high'] =  boot_event_early_stop_agg[boot_cols].quantile(0.975, axis=1)

boot_event_early_stop_agg = boot_event_early_stop_agg.reset_index()

#%%
#aggregate to event level
event_early_stops = pd.DataFrame(final_df.groupby(['event_id',
                                                   'event_name',
                                                   'event_date',
                                                   'partition'])[['early_stop',
                                                                  'pred_model1']].sum()).reset_index()

event_early_stops  = event_early_stops.merge(boot_event_early_stop_agg, on='event_id', how='left')

event_early_stops['actual_early_stop_within_range'] = np.nan
event_early_stops['actual_early_stop_within_range'] = np.where((event_early_stops['ci_low']<=event_early_stops['early_stop']) &
                                                               (event_early_stops['early_stop']<=event_early_stops['ci_high']),1,0)


print('coverage attempt 4: ', event_early_stops.groupby('partition')['actual_early_stop_within_range'].mean())


#%%
ci_logistic = event_early_stops[['event_id', 'event_name', 'event_date', 'partition',
                                 'early_stop','pred_model1', 'ci_low','ci_high']]

ci_logistic.to_csv(clean_data_path+'ci_logistic.csv', index=False)

#%%
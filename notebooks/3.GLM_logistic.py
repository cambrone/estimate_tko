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

################################################################
# PREPARE POSSIBLE PREDICTORS AND DEFINE TARGET
################################################################
# sort possible predictors
num_pred =  sorted(final_df.columns[(final_df.columns.str.endswith('_A_to_B_diff')) | 
                                    (final_df.columns.str.endswith('_abs_A_minus_B')) | 
                                    (final_df.columns.str.endswith('_A_plus_B')) |
                                    (final_df.columns.str.endswith('_A_plus_B_per_total_fights'))])

#create one hot encoded variables 
cat_pred = ['weight_class',	'gender']
cat_pred_encoded = []

for var in cat_pred:
    final_df[var] = final_df[var].str.lower()
    final_df[var] = final_df[var].str.replace(' ','_')

    one_hot = pd.get_dummies(final_df[var])
    cat_pred_encoded = cat_pred_encoded + list(one_hot.columns)
    final_df = final_df.join(one_hot)

#remove base levels multicollinearity
cat_pred_encoded.remove("f")
cat_pred_encoded.remove("women's_strawweight")

possible_pred = num_pred + cat_pred_encoded

################################ 
# PARTITION DATA 
################################
# define datasets to be standardized 
train_df = final_df[final_df['partition']=='T'].copy()
val_df = final_df[final_df['partition']=='V'].copy()
train_val_df = final_df[final_df['partition']!='H'].copy() 

no_debut = final_df[(final_df['debut_A']==0) & (final_df['debut_B']==0)].copy()


#%%
########################
# Explore predictors
########################
predictor_list = [
    'unanimous_decision_per_round_abs_A_minus_B',
    'early_stop_A_plus_B_per_total_fights',
    'fighter_reach_cm_A_plus_B',
    'switch_A_plus_B',
    'decision_A_plus_B',
    ]

for col in predictor_list:
    print('*********'+col+'*********')
    if len(final_df[col].unique())<5:
        continue
    else:
        
        summary_stats(df = final_df, col = col)
        
        #variable changes overtime
        change_over_time(df = final_df, col = col , time_col = 'month_year')

        #variable changes overtime by early decsion 
        change_over_time_by_target(df=final_df, target = 'early_stop', col = col, time_col='month_year')

        #early decision over buckets of variable
        exploratory_lift_chart(df = final_df, target = 'early_stop', col = col, nbins = 5)

        #early decision over buckets of variable no debut
        exploratory_lift_chart(df = no_debut, target = 'early_stop', col = col, nbins = 5)

#%%
model1 = sm.GLM(train_df[target], sm.add_constant(train_df[predictor_list]), family=sm.families.Binomial()).fit()

print(model1.summary())

train_val_df = score_w_model(df=train_val_df,
                                target= target,
                                predictor_list = predictor_list,
                                model_version = model1,
                                model_name = 'model1')

train_val_df.groupby('partition').apply(perf_metrics, target, 'pred_model1', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=train_val_df, 
           obs=target, 
           pred='pred_model1', 
           partition='partition',
           count='val',
           nbin=10)


#%%
# varibale importance    
var_importance_plot(model_version = model1)

#%%
# functional form with partial dependence plots
partial_resid_plot(df = train_val_df, 
                   predictor_list = predictor_list, 
                   work_resid = 'resid_working_model1',
                   model_version = model1, 
                   lowess_frac=0.3)

#%%       
factor_plot(df = train_val_df,
            obs = target,
            pred = 'pred_model1',
            count = 'val', 
            partition ='partition', 
            model_version = model1,
            predictor_list = predictor_list,
            factor=False)  

#%%
##################
# SCORE holdout 
##################
final_df = score_w_model(df=final_df,
                         target= target,
                         predictor_list = predictor_list,
                         model_version = model1,
                         model_name = 'model1')

final_df.groupby('partition').apply(perf_metrics, target, 'pred_model1', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=final_df, 
           obs=target, 
           pred='pred_model1', 
           partition='partition',
           count='val',
           nbin=10)

#%%
# functional form with partial dependence plots
partial_resid_plot(df = final_df, 
                   predictor_list = predictor_list, 
                   work_resid = 'resid_working_model1',
                   model_version = model1, 
                   lowess_frac=0.3)

#%%       
factor_plot(df = final_df,
            obs = target,
            pred = 'pred_model1',
            count = 'val', 
            partition ='partition', 
            model_version = model1,
            predictor_list = predictor_list,
            factor=False)  

# %%
###################### 
# Save model as pickle
######################
with open(model_path+"logistic_model.pkl", "wb") as f:
    pickle.dump(model1, f)

# %%

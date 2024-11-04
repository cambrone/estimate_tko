#%%
import pandas as pd
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import random 


from datetime import datetime
from config import clean_data_path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.interpolate import make_interp_spline, BSpline
from scipy.special import expit

from utils.model_eval import score_w_model, partial_resid_plot, calculate_nested_f_statistic, lift_chart, var_importance_plot, factor_plot, perf_metrics
np.random.seed(222)
pd.set_option('display.max_columns', None)


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
train_df_sd = final_df[final_df['partition']=='T'].copy()
val_df_sd = final_df[final_df['partition']=='V'].copy()
train_val_df_sd = final_df[final_df['partition']!='H'].copy() 
final_df_sd = final_df.copy()


################################ 
# STANDARDIZE NUMERIC PREDICTORS 
################################
#fit on training and apply on other datasets
scaler = StandardScaler()
scaler.fit(train_df_sd[num_pred])

train_df_sd[num_pred] = scaler.transform(train_df_sd[num_pred])
val_df_sd[num_pred] = scaler.transform(val_df_sd[num_pred])
train_val_df_sd[num_pred] = scaler.transform(train_val_df_sd[num_pred])
final_df_sd[num_pred] = scaler.transform(final_df_sd[num_pred])


########################################
# SCREEN POSSIBLE PREDICTORS USING LASSO
########################################
coefficients = pd.DataFrame(train_df_sd[possible_pred].columns,  columns=['features'])
metrics_loss = pd.DataFrame()
metrics_avg_pr = pd.DataFrame()
metrics_auc = pd.DataFrame()

penalty_list = [0.001,
                0.005,0.01, 0.04, 0.05,
                0.1,0.5,1,2]

#add in auc
for penalty in penalty_list:
    print(penalty)
    
    #fit and score model 
    model = LogisticRegression(random_state=0,
                        penalty='l1',
                        C=penalty,
                        solver='liblinear',
                        fit_intercept=True)
    
    model = model.fit(train_df_sd[possible_pred], train_df_sd[target])
    pred_train = model.predict_proba(train_df_sd[possible_pred])
    pred_val = model.predict_proba(val_df_sd[possible_pred])
    
    #extract coefficients
    coefficients = pd.concat([coefficients,
                              pd.DataFrame(np.transpose(model.coef_), columns=['penalty_' + str(penalty)])], axis = 1)

    # calculate performance metrics (loss, avg pr, auc)
    # metrics training
    train_loss = log_loss(y_true = train_df_sd[target].values, y_pred = pred_train[:,1])
    train_avg_p  = average_precision_score(y_true = train_df_sd[target].values, y_score = pred_train[:,1])
    train_auc = roc_auc_score(y_true = train_df_sd[target].values, y_score = pred_train[:,1])

    # metrics validation
    val_loss = log_loss(y_true = val_df_sd[target].values, y_pred = pred_val[:,1])
    val_avg_p = average_precision_score(y_true = val_df_sd[target].values, y_score = pred_val[:,1])
    val_auc = roc_auc_score(y_true = val_df_sd[target].values, y_score = pred_val[:,1])


    metrics_loss = pd.concat([metrics_loss,
                              pd.DataFrame([(penalty, train_loss, val_loss)],
                                           columns=['penalty', 'T', 'V'])], axis=0)
    
    metrics_avg_pr = pd.concat([metrics_avg_pr, 
                                pd.DataFrame([(penalty, train_avg_p, val_avg_p)],
                                             columns=['penalty', 'T', 'V'])], axis=0)
    
    metrics_auc = pd.concat([metrics_auc, 
                             pd.DataFrame([(penalty, train_auc, val_auc)],
                                          columns=['penalty', 'T', 'V'])], axis=0)



#%%
########################################
# CHECK LASSO SELECTED PREDICTORS
########################################
#based on above results both avg_pr and loss suggest variables with coefficients at penalty are important
#based on their coefficient since they are standardized we can use cofficient as measure of importance
penalty_lvl = str(0.1)
print('non-zero coeff: ', coefficients[coefficients['penalty_'+penalty_lvl]!=0].shape[0]/coefficients.shape[0])

feat_imp = coefficients[coefficients['penalty_'+penalty_lvl]!=0][['features', 'penalty_'+penalty_lvl]].copy()
feat_imp['abs_coeff'] = np.abs(feat_imp['penalty_'+penalty_lvl])
feat_imp = feat_imp.sort_values('abs_coeff', ascending=False)                      
feat_imp_plot = feat_imp.plot.bar(x='features', y='penalty_'+penalty_lvl, rot=90)
feat_imp_plot.set_title("feature importance lasso")
feat_imp_plot.set_xlabel("feature")
feat_imp_plot.set_ylabel("coeff")

#%%
lasso_pred = list(feat_imp['features'])
X = add_constant(train_df_sd[lasso_pred])
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns).to_frame()
vif['features'] = vif.index
vif = vif.rename(columns={0:'vif'})
vif.sort_values('vif', ascending=False)
vif_plot = vif.plot.bar(x='features', y='vif', rot=90)
vif_plot.set_title("vif")
vif_plot.set_xlabel("feature")
vif_plot.set_ylabel("coeff")

#%%
#####################################################
# Assess stability of LASSO based predictors
#####################################################
bootstrap_coef = pd.DataFrame(train_df_sd[possible_pred].columns,  columns=['features'])

for i in range(0,100):
    
    n_train = train_df_sd.shape[0]
    picks = np.random.randint(0,n_train,n_train)
    bootstrap_df = train_df_sd.iloc[picks]
    
    #fit and score model 
    model = LogisticRegression(random_state=0,
                        penalty='l1',
                        C=np.float(penalty_lvl),
                        solver='liblinear',
                        fit_intercept=True)
    
    bootstrap_model = model.fit(bootstrap_df[possible_pred], bootstrap_df[target])
    bootstrap_coef = pd.concat([bootstrap_coef,
                                pd.DataFrame(np.transpose(bootstrap_model.coef_), columns=['iter_' + str(i)])], axis = 1)

    if (i % 100) == 0:
        print('iter_'+str(i))

#%%
bootstrap_coef_1= bootstrap_coef.copy()
bootstrap_coef_1 = bootstrap_coef_1.astype(bool).astype(int)
bootstrap_coef_1['prop_feat'] = bootstrap_coef_1.mean(axis=1)
bootstrap_coef_1['features'] = bootstrap_coef['features']
bootstrap_coef_1 = bootstrap_coef_1.sort_values('prop_feat', ascending=False)
bootstrap_coef_1[['features', 'prop_feat']].head(50)

bootstrap_feat = bootstrap_coef_1[bootstrap_coef_1['prop_feat']>=0.75]['features']
#%%
##########################################################
# MODEL 1: STEPWISE SELECTION BASED ON AIC, AUC, AVG_PR
##########################################################
#based on these coefficients run a series of step wise variable selection using AIC and AUC
total_pred = list(bootstrap_feat)
accepted_pred_model1 = []

model1_aic = 1e9
model1_auc = 0

model1 = sm.GLM(train_df_sd[target], train_df_sd['val'], family=sm.families.Binomial()).fit()

#add in avg_pr, log_loss, auc on validation
for pred in total_pred: 
    print('********' + pred +'********')

    attempted_pred = accepted_pred_model1 + [pred]
    
    temp_model = sm.GLM(train_df_sd[target], sm.add_constant(train_df_sd[attempted_pred]), family=sm.families.Binomial()).fit()
    temp_pred_train = temp_model.predict(sm.add_constant(train_df_sd[attempted_pred]))
    temp_pred_val = temp_model.predict(sm.add_constant(val_df_sd[attempted_pred]))

    temp_aic = temp_model.aic
    temp_rsqr = temp_model.pseudo_rsquared()
    chisq_test = calculate_nested_f_statistic(model1, temp_model)
    
    temp_auc_train = roc_auc_score(train_df_sd[target], temp_pred_train)
    temp_auc_val = roc_auc_score(val_df_sd[target], temp_pred_val)
    
    temp_avg_pr_train = average_precision_score(train_df_sd[target], temp_pred_train)
    temp_avg_pr_val = average_precision_score(val_df_sd[target], temp_pred_val)

    print('aic model w '+pred+':' , temp_aic)
    print('Chi-sq test with '+ pred + ':', chisq_test)
    print('auc, T, V:', np.round(temp_auc_train,3),  np.round(temp_auc_val,3))
    print('pr, T, V:', np.round(temp_avg_pr_train,3),  np.round(temp_avg_pr_val,3))

    if (temp_aic < model1_aic) and (temp_auc_val > model1_auc):
        accepted_pred_model1 = attempted_pred
        
        model1_aic = temp_aic
        model1_auc = temp_auc_val   
        
        model1 = temp_model
        
        print(pred + 'improved model')
        print('best predictors: ', accepted_pred_model1)
        print('')

    else:
        print('adding ' + pred + ' did not lower aic')
        print('')    
        
#%%
#########################
# EVALUATE MODEL1
#########################
# score
train_val_df_sd = score_w_model(df=train_val_df_sd,
                                target= target,
                                predictor_list = accepted_pred_model1,
                                model_version = model1,
                                model_name = 'model1')
print(model1.summary())

print(train_val_df_sd.groupby('partition').apply(perf_metrics, target, 'pred_model1', 'val').reset_index())
#%%
# lift chart 
lift_chart(df=train_val_df_sd, 
           obs=target, 
           pred='pred_model1', 
           partition='partition',
           count='val',
           nbin=10)

#%%
# varibale importance    
var_importance_plot(model_version = model1)

# functional form with partial dependence plots
partial_resid_plot(df = train_val_df_sd, 
                   predictor_list = accepted_pred_model1, 
                   work_resid = 'resid_working_model1',
                   model_version = model1, 
                   lowess_frac=0.3)

#%%         
factor_plot(df = train_val_df_sd,
            obs = target,
            pred = 'pred_model1',
            count = 'val', 
            partition ='partition', 
            model_version = model1,
            predictor_list = accepted_pred_model1)            

#%%
##########################################################
# MODEL 2: Drop variables where pattern is unclear or correlated
##########################################################
predictor_list_model2 = accepted_pred_model1.copy()
predictor_list_model2.remove('fighter_height_cm_A_plus_B')
predictor_list_model2.remove('dq_per_round_abs_A_minus_B')
predictor_list_model2.remove('wins_A_to_B_diff')
predictor_list_model2.remove('num_rounds_per_total_fights_A_plus_B')
predictor_list_model2.remove('welterweight')
predictor_list_model2.remove('title_fight_abs_A_minus_B')
predictor_list_model2.remove('switch_A_plus_B')


model2 = sm.GLM(train_df_sd[target], sm.add_constant(train_df_sd[predictor_list_model2]), family=sm.families.Binomial()).fit()
chisq_test = calculate_nested_f_statistic(model1, model2)

print(model2.summary())
#%%

train_val_df_sd = score_w_model(df=train_val_df_sd,
                                target= target,
                                predictor_list = predictor_list_model2,
                                model_version = model2,
                                model_name = 'model2')

train_val_df_sd.groupby('partition').apply(perf_metrics, target, 'pred_model2', 'val').reset_index()


#%%
# lift chart 
lift_chart(df=train_val_df_sd, 
           obs=target, 
           pred='pred_model2', 
           partition='partition',
           count='val',
           nbin=10)

#%%

train_val_df_sd['debut_sum'] = train_val_df_sd['debut_A']+ train_val_df_sd['debut_B']
train_df_sd['debut_sum'] = train_df_sd['debut_A']+ train_df_sd['debut_B']

predictor_list_model3 = predictor_list_model2.copy()
predictor_list_model3 =  predictor_list_model3 + ['debut_sum']

model3 = sm.GLM(train_df_sd[target], sm.add_constant(train_df_sd[predictor_list_model3]), family=sm.families.Binomial()).fit()
chisq_test = calculate_nested_f_statistic(model1, model2)

print(model3.summary())

train_val_df_sd = score_w_model(df=train_val_df_sd,
                                target= target,
                                predictor_list = predictor_list_model3,
                                model_version = model3,
                                model_name = 'model3')

train_val_df_sd.groupby('partition').apply(perf_metrics, target, 'pred_model3', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=train_val_df_sd, 
           obs=target, 
           pred='pred_model3', 
           partition='partition',
           count='val',
           nbin=10)

factor_plot(df = train_val_df_sd,
            obs = target,
            pred = 'pred_model3',
            count = 'val', 
            partition ='partition', 
            model_version = model3,
            predictor_list = predictor_list_model3)   

#%%
#######MODEL 4
# train_val_df_sd['fighter_reach_cm_A_plus_B_male'] = train_val_df_sd['fighter_reach_cm_A_plus_B']* train_val_df_sd['m']
# train_df_sd['fighter_reach_cm_A_plus_B_male'] = train_df_sd['fighter_reach_cm_A_plus_B']* train_df_sd['m']

# predictor_list_model4 = predictor_list_model3.copy()
# predictor_list_model4 =  predictor_list_model4 + ['fighter_reach_cm_A_plus_B_male', 'm', 'debut_sum']

# model4 = sm.GLM(train_df_sd[target], sm.add_constant(train_df_sd[predictor_list_model4]), family=sm.families.Binomial()).fit()

# print(model4.summary())

# train_val_df_sd = score_w_model(df=train_val_df_sd,
#                                 target= target,
#                                 predictor_list = predictor_list_model4,
#                                 model_version = model4,
#                                 model_name = 'model4')

# train_val_df_sd.groupby('partition').apply(perf_metrics, target, 'pred_model4', 'val').reset_index()

# #%%
# # lift chart 
# lift_chart(df=train_val_df_sd, 
#            obs=target, 
#            pred='pred_model4', 
#            partition='partition',
#            count='val',
#            nbin=10)

# factor_plot(df = train_val_df_sd,
#             obs = target,
#             pred = 'pred_model4',
#             count = 'val', 
#             partition ='partition', 
#             model_version = model4,
#             predictor_list = predictor_list_model4)  

#%%
### Can the variubales be replaced fore mode intuitive alternative
predictor_list_model5 =  ['unanimous_decision_per_round_abs_A_minus_B',
                          'early_stop_A_plus_B_per_total_fights',
                          'fighter_reach_cm_A_plus_B',
                          'decision_A_plus_B']

model5 = sm.GLM(train_df_sd[target], sm.add_constant(train_df_sd[predictor_list_model5]), family=sm.families.Binomial()).fit()

print(model5.summary())

train_val_df_sd = score_w_model(df=train_val_df_sd,
                                target= target,
                                predictor_list = predictor_list_model5,
                                model_version = model5,
                                model_name = 'model5')

train_val_df_sd.groupby('partition').apply(perf_metrics, target, 'pred_model5', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=train_val_df_sd, 
           obs=target, 
           pred='pred_model5', 
           partition='partition',
           count='val',
           nbin=10)

factor_plot(df = train_val_df_sd,
            obs = target,
            pred = 'pred_model5',
            count = 'val', 
            partition ='partition', 
            model_version = model5,
            predictor_list = predictor_list_model5)  


#%%
###understand reversal group
train_val_df_sd['reversal'] = np.where((train_val_df_sd['pred_model5']>0.475) & 
                                       (train_val_df_sd['pred_model5']<=0.501),1,0)

reversal = train_val_df_sd[train_val_df_sd['reversal']==1][num_pred].describe()
reversal = reversal.add_suffix('_reverse')

non_reversal = train_val_df_sd[train_val_df_sd['reversal']==0][num_pred].describe()
non_reversal = non_reversal.add_suffix('_nonreverse')

combined = pd.concat([reversal, non_reversal], axis=1)
combined['stats']= combined.index
combined = combined[~combined['stats'].isin(['count', 'mean', 'std'])]

diff_df = pd.DataFrame()
for var in num_pred:
    print(var)
    combined[var+'_diff'] = np.abs(combined[var+'_reverse'] / combined[var+'_nonreverse'])
    combined[var+'_diff'].sum()
    
    diff_df = pd.concat([diff_df,
                         pd.DataFrame([[var, combined[var+'_diff'].sum()]],columns=['feat', 'diff'])], axis = 0)

# %%
#incorporate some of these differences
predictor_list_model6 =  ['num_rounds_per_total_fights_A_to_B_diff',
                          'unanimous_decision_per_round_abs_A_minus_B',
                          'early_stop_A_plus_B_per_total_fights',
                          'fighter_reach_cm_A_plus_B',
                          'decision_A_plus_B']

model6 = sm.GLM(train_df_sd[target], sm.add_constant(train_df_sd[predictor_list_model6]), family=sm.families.Binomial()).fit()

print(model6.summary())

train_val_df_sd = score_w_model(df=train_val_df_sd,
                                target= target,
                                predictor_list = predictor_list_model6,
                                model_version = model6,
                                model_name = 'model6')

train_val_df_sd.groupby('partition').apply(perf_metrics, target, 'pred_model6', 'val').reset_index()

#%%
# lift chart 
lift_chart(df=train_val_df_sd, 
           obs=target, 
           pred='pred_model6', 
           partition='partition',
           count='val',
           nbin=10)

factor_plot(df = train_val_df_sd,
            obs = target,
            pred = 'pred_model6',
            count = 'val', 
            partition ='partition', 
            model_version = model6,
            predictor_list = predictor_list_model6)  

# %%

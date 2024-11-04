import pandas as pd
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from scipy.special import expit
from sklearn.metrics import log_loss, average_precision_score,roc_auc_score, mean_squared_error


def score_w_model(df, target, predictor_list, model_version, model_name):
    df['pred_'+model_name] = model_version.predict((sm.add_constant(df[predictor_list])))
    df['pred_linear_'+model_name] = model_version.predict(sm.add_constant(df[predictor_list]), linear=True)
    df['resid_raw_'+model_name] = df[target] - df['pred_'+model_name]
    df['resid_working_'+model_name] = df['resid_raw_'+model_name]*(1/ (df['pred_'+model_name] * (1 - df['pred_'+model_name])))
    
    return df

def partial_resid_plot(df, predictor_list, work_resid, model_version, lowess_frac=0.3):

    for var in predictor_list:
        df['comp_res'] = df[work_resid] + model_version.params[var]*df[var]
        
        for part in df['partition'].unique():
            temp = df[df['partition']==part].copy()
            
            fig, ax1 = plt.subplots(figsize=(10, 10))
            ax1.scatter(temp[var], temp['comp_res'], s=2, alpha=0.5)
            ax1.plot(temp[var], model_version.params[var]*temp[var], '-', color='tab:blue')
            
            smoothed = sm.nonparametric.lowess(exog=temp[var], endog=temp['comp_res'], frac=lowess_frac)
            ax1.plot(smoothed[:, 0], smoothed[:, 1],  '-', color='tab:orange')

            ax1.tick_params(axis='x', labelrotation=90)  
            
            plt.title(var+' - '+part)
            ax1.set_xlabel(var)
            ax1.set_ylabel('comp_res')
            ax1.set_xticklabels(temp[var],  rotation=45)
            
            plt.show()


def calculate_nested_f_statistic(small_model, big_model):
    """Given two fitted GLMs, the larger of which contains the parameter space of the smaller, return the F Stat and P value corresponding to the larger model adding explanatory power"""
    addtl_params = big_model.df_model - small_model.df_model
    f_stat = (small_model.deviance - big_model.deviance) / (addtl_params * big_model.scale)
    df_numerator = addtl_params
    # use fitted values to obtain n_obs from model object:
    df_denom = (big_model.fittedvalues.shape[0] - big_model.df_model)
    p_value = stats.f.sf(f_stat, df_numerator, df_denom)
    return (f_stat, p_value)

def perf_metrics(df, obs, pred, cnt):
    fight_cnt = df[cnt].sum()
    obs_prop_pos = df[obs].mean()
    pred_prop_pos = df[pred].mean()

    loss = log_loss(y_true = df[obs].values, y_pred = df[pred].values)
    avg_p  = average_precision_score(y_true = df[obs].values, y_score = df[pred].values)
    auc = roc_auc_score(y_true = df[obs].values, y_score =  df[pred].values)
    
    return pd.Series(dict(fight_cnt = fight_cnt, 
                          obs_prop_pos= obs_prop_pos,
                          pred_prop_pos=pred_prop_pos,
                          loss = loss, 
                          avg_p = avg_p,
                          auc = auc))
    
def perf_metrics_poisson(df, obs, pred, cnt):
    fight_cnt = df[cnt].sum()
    obs_prop_pos = df[obs].mean()
    pred_prop_pos = df[pred].mean()

    rmse = mean_squared_error(y_true = df[obs].values,
                              y_pred = df[pred].values)

    return pd.Series(dict(fight_cnt = fight_cnt, 
                          obs_prop_pos= obs_prop_pos,
                          pred_prop_pos=pred_prop_pos,
                          rmse = rmse))
    
def lift_chart(df, obs, pred, partition, count, nbin):

    ser, bins = pd.qcut(df[df[partition]=='T'][pred], 10, retbins=True, duplicates='drop')
    df[pred+'_grp'] = pd.cut(df[pred], bins=bins)
    df[pred+'_grp'] =df[pred+'_grp'].astype(str)
    df[pred+'_grp'] = np.where(df[pred+'_grp']=='nan','missing', df[pred+'_grp'])

    for part in df[partition].unique():
        temp = df[(df[partition]==part) & (df[pred+'_grp']!='missing')].copy()

        cnt = pd.DataFrame(temp.groupby([pred+'_grp'])[count].sum()).reset_index()
        mean = pd.DataFrame(temp.groupby([pred+'_grp'])[[pred,obs]].mean()).reset_index()
        temp = cnt.merge(mean, on = [pred+'_grp'], how='left')
        
        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax2 = ax1.twinx()
        
        temp[count].plot(kind='bar', ax=ax2, alpha = 0.15)
        temp[pred].plot(kind='line', marker='d', ax=ax1, color='red')
        temp[obs].plot(kind='line', marker='.', ax=ax1, color='blue')

        plt.title(part)
        ax1.set_xlabel(pred+'_grp')
        ax1.set_ylabel(obs)
        ax2.set_ylabel(count)
        ax1.set_xticklabels(temp[pred+'_grp'],  rotation=45)

        plt.show()
        
def var_importance_plot(model_version):
    temp = pd.DataFrame(model_version.params).reset_index()
    temp = temp.rename(columns={'index':'features', 0:'coeff'})
    temp['abs_coeff'] = np.abs(temp['coeff'])
    temp = temp.sort_values('abs_coeff', ascending=False)         
    
    fig, ax1 = plt.subplots(figsize=(10, 10))
    temp['coeff'].plot(kind='bar', ax=ax1, alpha = 0.15)
                 
    ax1.set_title("feature importance lasso")
    ax1.set_xlabel("feature")
    ax1.set_ylabel("coeff")
    ax1.set_xticklabels(temp['features'],  rotation=45)
    
    plt.show()
    
def factor_plot(df, obs, pred, count, partition, model_version, predictor_list, factor=True):
    
    for var in predictor_list:
        df['factor'] = expit(model_version.params['const'] + model_version.params[var]*df[var])
        
        df[var +'_temp'] =  df[var].replace(to_replace=0, value=np.nan)
        ser, bins = pd.qcut(df[df[partition]=='T'][var +'_temp'], 10, retbins=True, duplicates='drop')
        df[var+'_grp'] = pd.cut(df[var +'_temp'], bins=bins)
        df[var+'_grp'] =df[var+'_grp'].astype(str)
        df[var+'_grp'] = np.where(df[var+'_grp']=='nan','(0.0,0.0]', df[var+'_grp'])

        for part in df[partition].unique():
            temp = df[df[partition]==part].copy()

            cnt = pd.DataFrame(temp.groupby([var+'_grp'])[count].sum()).reset_index()
            mean = pd.DataFrame(temp.groupby([var+'_grp'])[[pred, 'factor', obs]].mean()).reset_index()
            temp = cnt.merge(mean, on = [var+'_grp'], how='left')
        
            fig, ax1 = plt.subplots(figsize=(10, 10))
            ax2 = ax1.twinx()
            
            temp[count].plot(kind='bar', ax=ax2, alpha = 0.15)
            temp[pred].plot(kind='line', marker='d', ax=ax1, color='red')
            if factor==True:
                temp['factor'].plot(kind='line', marker='d', ax=ax1, color='green')
            temp[obs].plot(kind='line', marker='.', ax=ax1, color='blue')

            plt.title(part)
            ax1.set_xlabel(var+'_grp')
            ax1.set_ylabel(obs)
            ax2.set_ylabel(count)
            ax1.set_xticklabels(temp[var+'_grp'],  rotation=45)
            
            ax1.legend(loc="upper left")

            plt.show()
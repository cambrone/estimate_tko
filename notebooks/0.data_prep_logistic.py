#%%
import pandas as pd
import numpy as np 
from datetime import datetime
import random 

np.random.seed(222)

#%%
pd.set_option('display.max_columns', None)

# add in paperview indicator
# create training, validation and oot
# aggregate and create poisson file

###################################################
# Possible features: last 3 fight average against all career (are they on a run)
###################################################
#%%
from config import raw_data_path, clean_data_path
from utils.data_prep import clean_id_cols

######################################
# Read in fight stats
######################################
fight_stats = pd.read_csv(raw_data_path+"ufc_fight_stat_data.csv")
fight_stats = fight_stats.drop('fight_url', axis=1)

# missing fighter ids due to names not following first/last name convention
print(fight_stats['fighter_id'].isnull().sum())
fight_stats = fight_stats[~fight_stats['fighter_id'].isnull()]

#clean columns
id_columns = ['fight_stat_id', 'fight_id', 'fighter_id']
fight_stats = clean_id_cols(df = fight_stats, id_cols = id_columns)

for id_col in id_columns:
    fight_stats[id_col+'_num'] = fight_stats[id_col].astype(int)

fight_stats['ctrl_time'] = fight_stats['ctrl_time'].str.strip()
fight_stats['ctrl_time'] = np.where(fight_stats['ctrl_time']=='--','0:00',fight_stats['ctrl_time'])
temp = fight_stats['ctrl_time'].str.split(':',expand=True)
temp = temp.rename(columns={0:'minutes',
                            1: 'seconds'})
temp[['minutes', 'seconds']] = temp[['minutes', 'seconds']].astype(float)                        
temp['minutes_in_secs'] = temp['minutes'] * 60
temp['ctrl_time'] = temp['minutes_in_secs'] + temp['seconds']

fight_stats = pd.concat([fight_stats, temp], axis=1)
#%%
fight_stats = fight_stats.sort_values(['fighter_id_num', 'fight_id_num'], ascending=True).reset_index()

######################################
# Read in athlete data
######################################
athlete = pd.read_csv(raw_data_path+"ufc_fighter_data.csv")
athlete = athlete.drop('fighter_url', axis=1)

id_columns = ['fighter_id']
athlete = clean_id_cols(df = athlete, id_cols = id_columns)

# clean names
athlete['fighter_l_name'] = np.where(athlete['fighter_l_name'].isnull(), 'missing', athlete['fighter_l_name'])
athlete['fighter_f_name'] = athlete['fighter_f_name'].str.lower().replace(' ', '_')
athlete['fighter_l_name'] = athlete['fighter_l_name'].str.lower().replace(' ', '_')

athlete['fighter_name'] = athlete['fighter_f_name'] + '_' + athlete['fighter_l_name']

athlete['fighter_dob'] = pd.to_datetime(athlete['fighter_dob']) 

athlete['fighter_stance'] = athlete['fighter_stance'].str.lower()
athlete['fighter_stance'] = np.where(athlete['fighter_stance'].isnull(), 'orthodox',athlete['fighter_stance'])
stance = pd.get_dummies(athlete['fighter_stance'])
stance.columns = stance.columns.str.replace(' ','_')
athlete = pd.concat([athlete, stance], axis=1)  

athlete['fighter_height_cm'] = np.where(athlete['fighter_height_cm'].isnull(), athlete['fighter_height_cm'].mean(), athlete['fighter_height_cm'])
athlete['fighter_reach_cm'] = np.where(athlete['fighter_reach_cm'].isnull(), athlete['fighter_reach_cm'].mean(), athlete['fighter_reach_cm'])

######################################
# Read in fight outcome
######################################
outcome = pd.read_csv(raw_data_path+"ufc_fight_data.csv")
outcome = outcome.drop('fight_url', axis=1)

#clean
id_columns = ['fight_id','event_id','f_1','f_2', 'winner']
outcome = clean_id_cols(df = outcome, id_cols = id_columns)
outcome['result'] = outcome['result'].str.lower()
outcome['result_details'] = outcome['result_details'].str.lower()

outcome['num_rounds'] = outcome['num_rounds'].str.replace(r'\D+', '0')
outcome['num_rounds'] = outcome['num_rounds'].astype(float)

outcome['num_rounds'] = outcome['num_rounds'].astype(float)


#features
outcome['title_fight'] = np.where(outcome['title_fight']=='T',1,0)

outcome['decision'] = np.where(outcome['result'].isin(['decision']),1,0)
outcome['ko'] = np.nan
outcome['ko'] = np.where(outcome['result'].isin(['ko/tko',
                                                "tko - doctor's stoppage"]),1,0)
outcome['submission'] = np.where(outcome['result'].isin(['submission']),1,0)
outcome['dq'] = np.where(outcome['result'].isin(['dq']),1,0)

# combined result
outcome['early_stop'] = np.nan
outcome['early_stop'] = np.where(~outcome['result'].isin(['decision','dq']),1,0)

outcome['unanimous_decision'] = np.nan
outcome['unanimous_decision'] = np.where((outcome['decision']==1) & 
                                         (outcome['result_details']=='unanimous'),1,0)

#%%
######################################
# Read in event data
######################################
event = pd.read_csv(raw_data_path+"ufc_event_data.csv")
event = event.drop('event_url', axis=1)

id_columns = ['event_id']
event = clean_id_cols(df = event, id_cols = id_columns)
event['event_date'] = pd.to_datetime(event['event_date'])


#%%
####################################
# Merge files together
####################################
#merge fight and athlete data
print('before athlete merge',fight_stats.shape[0])
all_data = fight_stats.merge(athlete[['fighter_id', 
                                        'fighter_name',
                                        'fighter_height_cm',
                                        'fighter_reach_cm',	
                                        'open_stance',
                                        'orthodox',
                                        'sideways',
                                        'southpaw',
                                        'switch',
                                        'fighter_dob']], on ='fighter_id', how='left')
print('after athlete merge', all_data.shape[0])

#merge fight and outcome data
print('before outcome merge',all_data.shape[0])
all_data = all_data.merge(outcome, on = 'fight_id', how='left')
print('before after merge',all_data.shape[0])

#merge fight and outcome data
print('before event merge',all_data.shape[0])
all_data = all_data.merge(event, on = 'event_id', how='left')
print('before after merge',all_data.shape[0])

all_data['total_fights'] = 1

all_data['wins'] = np.nan 
all_data['wins'] = np.where(all_data['fighter_id'] == all_data['winner'],1,0)

all_data['losses'] = np.nan 
all_data['losses'] = np.where(all_data['fighter_id'] != all_data['winner'],1,0)

all_data['age'] =  (all_data['event_date'] - all_data['fighter_dob']).dt.days
all_data['age'] = np.where(all_data['age'].isnull(),365*25,all_data['age'])

#keep data for relevant time period (modern ufc)
all_data['year'] = pd.to_datetime(all_data['event_date']).dt.year
all_data['month_year'] = all_data['event_date'].dt.to_period('M')


#%%
#####################################################
# Base Dataframe 
#####################################################
fight_fighter_ids = all_data[['fight_id', 'fighter_id', 'fight_id_num','fighter_name']].copy()

#####################################################
# Fighter age, height, stance, fighter_reach_cm
####################################################
print(fight_fighter_ids.shape[0])
fight_fighter_ids = fight_fighter_ids.merge(all_data[['fight_id', 'fighter_id', 
                                                    'age','fighter_height_cm','fighter_reach_cm',
                                                    'open_stance','orthodox','sideways','southpaw','switch']],
                                            on = ['fight_id', 'fighter_id'],
                                            how='left')
print(fight_fighter_ids.shape[0])

#%%
#####################################################
# for each fighter keep data up to current fight 
#####################################################
#career stats
career_stats_vars =['knockdowns','total_strikes_att',
                    'total_strikes_succ','sig_strikes_att',
                    'sig_strikes_succ','takedown_att',
                    'takedown_succ','submission_att',
                    'reversals','ctrl_time','num_rounds',
                    'title_fight', 'decision','ko','submission' ,      
                    'dq','early_stop','unanimous_decision', 
                    'wins','losses','total_fights']

fight_before_event =  all_data[['fighter_id', 'fight_id_num', 'fighter_id_num', *career_stats_vars]].copy()
fight_before_event =  fight_before_event.rename(columns= {'fight_id_num':'fight_id_before'})

career_stats = fight_fighter_ids.merge(fight_before_event, on='fighter_id', how='left')
career_stats = career_stats[career_stats['fight_id_num'] > career_stats['fight_id_before']].copy()

summary_career_stats = pd.DataFrame(career_stats.groupby(['fight_id', 'fighter_id'])[career_stats_vars].sum()).reset_index()
#%%
# for fighters first fight metrics, take average fight before event
average_career_stats =  all_data[['fight_id', 'fight_id_num', *career_stats_vars]].copy()
average_career_stats = pd.DataFrame(average_career_stats.groupby(['fight_id','fight_id_num'])[career_stats_vars].mean()).reset_index()
average_career_stats = average_career_stats.sort_values('fight_id_num')
#%%
for var in career_stats_vars:
    average_career_stats[var+'_mean'] = average_career_stats[var].expanding().mean()
    average_career_stats = average_career_stats.drop(var, axis=1)

average_career_stats = average_career_stats.drop('fight_id_num', axis=1)

print(fight_fighter_ids.shape[0])
fight_fighter_ids = fight_fighter_ids.merge(summary_career_stats, on=['fight_id', 'fighter_id'], how='left' )
fight_fighter_ids = fight_fighter_ids.merge(average_career_stats, on=['fight_id'], how='left' )
print(fight_fighter_ids.shape[0])

#%%
for var in career_stats_vars:
    fight_fighter_ids['debut'] = np.where(fight_fighter_ids[var].isnull(), 1, 0)
    fight_fighter_ids[var] = np.where(fight_fighter_ids[var].isnull(),  
                                        fight_fighter_ids[var+'_mean'],
                                        fight_fighter_ids[var])
    fight_fighter_ids = fight_fighter_ids.drop(var+'_mean', axis=1)

#%%
#############################################################
# Feature engineering at fight_fighter level
#############################################################
#normalized features by number of fights
for var in career_stats_vars:
    fight_fighter_ids[var+'_per_round'] = fight_fighter_ids[var]/fight_fighter_ids['num_rounds']

#success ratios
fight_fighter_ids['total_strikes_succ_ratio'] = fight_fighter_ids['total_strikes_succ']/fight_fighter_ids['total_strikes_att']
fight_fighter_ids['sig_strikes_succ_ratio'] = fight_fighter_ids['sig_strikes_succ']/fight_fighter_ids['sig_strikes_att']
fight_fighter_ids['takedown_succ_ratio'] = fight_fighter_ids['takedown_succ']/fight_fighter_ids['takedown_att']
fight_fighter_ids['submission_att_ratio'] = fight_fighter_ids['submission_att']/fight_fighter_ids['ctrl_time']
fight_fighter_ids['reversals_succ_ratio'] = fight_fighter_ids['reversals']/fight_fighter_ids['ctrl_time']

#normalize by number of fights
vars_to_norm = ['knockdowns', 'total_strikes_att', 'total_strikes_succ', 'sig_strikes_att',	
                'sig_strikes_succ', 'takedown_att',	'takedown_succ'	, 'submission_att',
                'reversals',	'ctrl_time'	, 'num_rounds', 'title_fight',	'decision',	
                'ko','submission', 'dq', 'early_stop','unanimous_decision','wins','losses']
for var in vars_to_norm:
    fight_fighter_ids[var+'_per_total_fights'] = fight_fighter_ids[var]/fight_fighter_ids['total_fights']

fight_fighter_ids =  fight_fighter_ids.replace([np.inf, -np.inf], 0)



#%%%
##############################################################
# create fight level dataset
##############################################################
#not all fights scraped correctly so not all have 1, 2
fight_fighter_ids['fight_order'] = fight_fighter_ids.groupby(['fight_id']).cumcount()+1

temp = pd.DataFrame(fight_fighter_ids.groupby('fight_id')['fight_order'].sum()).reset_index()
temp['missing_fighter'] = np.where(temp['fight_order']!=3,1,0)
temp = temp.drop('fight_order', axis=1)

fight_fighter_ids = fight_fighter_ids.merge(temp, on ='fight_id', how='left')
fight_fighter_ids = fight_fighter_ids[fight_fighter_ids['missing_fighter']==0].copy()

fighter_a = fight_fighter_ids[fight_fighter_ids['fight_order']==1].copy()
fighter_a = fighter_a.add_suffix('_A')
fighter_a = fighter_a.rename(columns={'fight_id_A':'fight_id',
                                    'fight_id_num_A':'fight_id_num'})

fighter_b = fight_fighter_ids[fight_fighter_ids['fight_order']==2].copy()
fighter_b = fighter_b.add_suffix('_B')
fighter_b = fighter_b.rename(columns={'fight_id_B':'fight_id',      
                                    'fight_id_num_B':'fight_id_num'})

print(fighter_a.shape[0])
final_fight_df = fighter_a.merge(fighter_b, on = ['fight_id'], how='left')
print(final_fight_df.shape[0])

#########################################
# Feature engineer fight level variables
###########################################
vars_fight = list(final_fight_df.columns)
vars_to_remove = ['fight_id','fighter_id_A','fight_id_num_x',
                'fighter_name_A','fighter_id_B','fight_id_num_y',
                'fighter_name_B','missing_fighter_A',
                'missing_fighter_B']

vars_fight = [var for var in vars_fight if var not in vars_to_remove]
vars_fight = [var.rstrip( "_A" ) for var in vars_fight]
vars_fight = [var.rstrip( "_B" ) for var in vars_fight]
vars_fight = list(set(vars_fight))

for var in vars_fight: 
    final_fight_df[var+'_abs_A_minus_B'] = np.abs(final_fight_df[var+'_A']-final_fight_df[var+'_B'])
    final_fight_df[var+'_A_plus_B'] = final_fight_df[var+'_A'] + final_fight_df[var+'_B']
    final_fight_df[var+'_A_to_B_diff'] = final_fight_df[var+'_abs_A_minus_B']/final_fight_df[var+'_A_plus_B']
    final_fight_df[var+'_A_to_B_diff'] = final_fight_df[var+'_A_to_B_diff'].replace([np.inf, -np.inf], np.nan)
    final_fight_df[var+'_A_to_B_diff'] = np.where(final_fight_df[var+'_A_to_B_diff'].isnull(),final_fight_df[var+'_A_to_B_diff'].mean(), final_fight_df[var+'_A_to_B_diff'])

#%%
#normalize by number of fights
vars_to_norm = ['knockdowns', 'total_strikes_att', 'total_strikes_succ', 'sig_strikes_att',	
                'sig_strikes_succ', 'takedown_att',	'takedown_succ'	, 'submission_att',
                'reversals',	'ctrl_time'	, 'num_rounds', 'title_fight',	'decision',	
                'ko','submission', 'dq', 'early_stop','unanimous_decision','wins','losses']
for var in vars_to_norm:
    final_fight_df[var+'_A_plus_B_per_total_fights'] = final_fight_df[var+'_A_plus_B']/final_fight_df['total_fights_A_plus_B']


#%%
### Add in fight outcome
print(final_fight_df.shape[0])
final_fight_df = final_fight_df.merge(outcome[['fight_id','event_id',
                                            'weight_class','gender',
                                            'early_stop',
                                            'decision',
                                            'ko']], on='fight_id', how='left')
print(final_fight_df.shape[0])

### event info
print(final_fight_df.shape[0])
final_fight_df = final_fight_df.merge(event, on='event_id', how='left')
print(final_fight_df.shape[0])


final_fight_df['val'] =1

#%%
# create partition based on event
# same partition will be used for poisson model
partition = final_fight_df[['event_id','event_name','event_date', 'val']].copy()
partition = partition.drop_duplicates()
partition['year'] = partition['event_date'].dt.year

print('before partition', partition.shape[0])
msk = np.random.rand(len(partition)) < 0.8
train = partition[msk].copy()
train['partition'] = 'T'
validation = partition[~msk].copy()
validation['partition'] = 'V'
partition = pd.concat([train,validation], axis=0)
print('after partition', partition.shape[0])

partition['partition'] = np.where(partition['year']>=2022, 'H', partition['partition'])



print('before partition', final_fight_df.shape[0])
final_fight_df = final_fight_df.merge(partition[['event_id','partition']], on='event_id', how='left')
print('after partition', final_fight_df.shape[0])


#%%
# Final check
final_fight_df[final_fight_df['fighter_id_B']=='3493'][['fight_id',
'fight_id_num_x',
'fighter_id_B',
'fighter_name_B',
'age_B',
'fighter_id_B',
'wins_B',
'losses_B',
'total_fights_B',
'fighter_name_A',
'fight_order_B',
'age_A',
'weight_class',
'gender',
'early_stop']].sort_values('fight_id_num_x', ascending=True)

final_fight_df = final_fight_df.fillna(0)

final_fight_df['year'] = final_fight_df['event_date'].dt.year
final_fight_df = final_fight_df[final_fight_df['year']>=2013].copy()

final_fight_df.to_csv(clean_data_path+'final_fight_df_binary.csv', index=False)
print(final_fight_df.groupby('partition')['early_stop'].mean())
#%%
''' 
    Helper functions for modelling notebooks.

    Please add functions here to use in the notebooks, then add to the import section
    and rerun the notebooks to use them.

    preprocessing, cross validation, feature importance, and model scoring functions are included here
'''

# data processing
import pandas as pd
import numpy as np
from collections import defaultdict

# modelling
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, PrecisionRecallDisplay, auc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
# visualisations
import matplotlib.pyplot as plt

import os
seed=42


def heidke_skill_score(y_true, y_pred, precision=4):
  '''
    Given prediction and ground truth labels, calculate Heidke Skill Score. to the indicated
    level of precision (default = 4). HSS is a commonly used forecasting skill metric, which compares
    the performance of the model to that of a referemce forecast.
    
    Usually, the reference is a system that makes random forecasts. It takes values between 1 and -1, with 
    values greater than 1 indicating the model is better than random prediction, and values below 0 indicating it is worse than random.
  '''
  cm = confusion_matrix(y_true, y_pred)
  n = cm.flatten().sum()
  tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
  chance_preds = ((tp + fp)*(tp + fn) + (fp + tn)*(fn + tn))/n
  hss = np.round((tp + tn - chance_preds)/(n - chance_preds), precision)

  return hss


def preprocess(X_train, X_test, cat_vars, num_vars, cat_encoder):
  '''
    Function to preprocess inputs to models. Data must be separated into train and test
    for one hot encoding to prevent data leakage.

    Arguments:
      X_train (pd.DataFrame): The model training dataset. This is used for fitting imputers, encoders, etc.
      X_test (pd.Dataframe): the test set for model.
      cat_encoder(str): categorical encoder options ('oh', 'oe'). 'oh' (OneHotencoding) is correct for xgboost so set as default.
      Alternative is ordinal encoding 'oe', which is just recognised as numeric column by xgboost.
  '''
  # different variable types have separate preprocesses
  X_train_cat, X_test_cat = X_train[cat_vars], X_test[cat_vars]
  X_train_num, X_test_num = X_train[num_vars], X_test[num_vars]

  # Apply encoder to categorical cols
  if cat_encoder=='oe':
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat_enc = pd.DataFrame(encoder.fit_transform(X_train_cat))
    X_test_cat_enc = pd.DataFrame(encoder.transform(X_test_cat))
    
    # restore variable names
    X_train_cat_enc.columns = encoder.get_feature_names_out()
    X_test_cat_enc.columns = encoder.get_feature_names_out()
  elif cat_encoder=='oh':
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat_enc = pd.DataFrame(encoder.fit_transform(X_train_cat))
    X_test_cat_enc = pd.DataFrame(encoder.transform(X_test_cat))
    
    # restore variable names
    X_train_cat_enc.columns = encoder.get_feature_names_out()
    X_test_cat_enc.columns = encoder.get_feature_names_out()
  else:
    raise ValueError('cat_enocder argument input is invalid!')
  # restore indices
  X_train_cat_enc.index, X_test_cat_enc.index = X_train.index, X_test.index 

  # numerical variables
  scaler = StandardScaler()
  X_train_num_enc = pd.DataFrame(scaler.fit_transform(X_train_num))
  X_test_num_enc = pd.DataFrame(scaler.transform(X_test_num))
  
  # restore feature names and indices
  X_train_num_enc.columns = scaler.get_feature_names_out()
  X_test_num_enc.columns = scaler.get_feature_names_out()
  X_train_num_enc.index, X_test_num_enc.index = X_train.index, X_test.index


  X_train_p = pd.concat([X_train_num_enc, X_train_cat_enc], axis=1)
  X_test_p = pd.concat([X_test_num_enc, X_test_cat_enc], axis=1)

  return X_train_p, X_test_p

def score_model(model, X_train, X_valid, y_train, y_valid, pre_X_valid):
  '''
    Score models in terms of our evaluation metrics. Currently, metrics include
    F1, Heidke skill score, and these metrics specifically for fog state transitions.
    Precision-Recall-AUC is also available for the probabilistic models. Precision-Recall curve plot is saved for probabilistic models

    Arguments:
      model: predictive model to be scored. (e.g., XGBoost, SVM)
      X_train, X_valid: training and validation predictors
      y_train, y_valid: training and validation target
      pre_X_valid: Original X_valid dataset before passing to the preprocess function. This allows us to store
      the fog transition indices for transition evaluation.

    Returns:
      2 Dictionaries, scores and plots (if any).
  '''

  scores = {}
  plots = {}
  model = model.fit(X_train, y_train)             
  y_pred = model.predict(X_valid)

  try:
    y_pred_proba = model.predict_proba(X_valid)[:,1]

    precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba)
    pr_auc = np.round(auc(recall, precision), 4)
    scores['pr_auc'] = pr_auc
      
    pr_plot = PrecisionRecallDisplay.from_estimator(model, X_valid, y_valid)
    pr_plot.ax_.set_title('Precision-Recall Curve')
    plots['pr_plot'] = pr_plot
  except:
    print("Unable to calculate probabilities of class labels, so AUC omitted. Check that the model has the predict_proba function.")

  cm = confusion_matrix(y_valid, y_pred)
  # Confusion Matrix format
  # TN FP
  # FN TP

  X_valid_p = X_valid.copy()
  y_valid_p = y_valid.copy()
  
  # add transiiton as a column after model fitting
  X_valid_p['transition'] = pre_X_valid['transition']
  X_valid_p['fog_state'] = pre_X_valid['fog_state']

  # align the indices of training data and target
  X_valid_p = X_valid_p.reset_index()
  y_valid_p = y_valid_p.reset_index(drop=True)

  # find indices of transitions, and the index right before it
  t_indices = X_valid_p.loc[X_valid_p['transition']==1].index
  before_t_indices = t_indices - 1
  t_indices = t_indices.append(before_t_indices)
  
  # check if transition indices actually show fog state transitions
  #print(X_valid_p.loc[t_indices[0]-1:t_indices[0]+2, ['fog_state', 'transition']])

  # assign these to new y variables
  y_pred_transition = y_pred[t_indices]
  y_valid_transition = y_valid_p.iloc[t_indices]

  # calculate F1 and Heidke Skill Scores
  f1 = np.round(f1_score(y_valid, y_pred)*100, 2)
  hss = heidke_skill_score(y_valid, y_pred)
  transition_f1 = np.round(f1_score(y_valid_transition, \
                                    y_pred_transition)*100, 2)
  transition_hss = heidke_skill_score(y_valid_transition, y_pred_transition)

  scores['f1'] = f1
  scores['hss'] = hss
  scores['transition_f1'] = transition_f1
  scores['transition_hss'] = transition_hss
  scores['confusion_matrix'] = cm

  return scores, plots


def manual_cross_validate(model, X, y, num_vars, cat_vars, cat_encoder, folds=5, calc_feature_importance=True, verbose=True):
  '''
    Perform cross validation on the train/valid set using time series split, and compute feature importances
    for each fold if calc_feature_importance=True.

    This function allows us to do preprocessing as part of the cross validation process.

    Arguments:
      model: predictive model that we are evaluating
      
      X, y: predictors and target dataframes
      
      num_vars (list<str>): list of names of numerical variables as strings. 
      Should be overridden if not using every variable in df_train, 
      or else will encounter KeyError.
      
      cat_vars (list<str>): similar to num_vars but for categorical vars. These need to be
      specified for the separate preprocesses.
      
      folds (int): number of folds for cv. default: 5
      
      calc_feature_importance (boolean): whether or not to calculate feature 
      importance in each fold. default: true

      verbose (boolean): whether to print intermediate results (fold number, confusion matrix, data shape). default: True

    Returns:
      3 collections, scores[0], feature importance[1], models[2]
  '''
  tss = TimeSeriesSplit(n_splits=folds)
  oof = pd.DataFrame()                # out-of-fold result
  models = []                         # models
  scores = defaultdict(list)           # validation score
  gain_importance_tables = []          # feature importance list
  
  for fold, (trn_idx, val_idx) in enumerate(tss.split(X, y)):

    print("Fold :", fold+1)

    # create dataset
    pre_X_train, y_train = X.loc[trn_idx], y[trn_idx]
    pre_X_valid, y_valid = X.loc[val_idx], y[val_idx]

    X_train, X_valid = preprocess(pre_X_train, pre_X_valid, num_vars=num_vars, cat_vars=cat_vars, cat_encoder=cat_encoder)

    model_scores, model_plots = score_model(model, X_train, X_valid, y_train, y_valid, pre_X_valid)

    
    # keep scores and models
    scores['f1_score'].append(model_scores['f1'])
    scores['heidke_skill_score'].append(model_scores['hss'])
    scores['transition_f1_score'].append(model_scores['transition_f1'])
    scores['transition_hss_score'].append(model_scores['transition_hss'])
    
    # probabilistic model scores. Won't save for deterministic models.
    if len(model_plots.keys()) > 0:
      scores['pr_auc'].append(model_scores['pr_auc'])

    models.append(model)

    if verbose:
      print("training size:", X_train.shape)
      print("test size:", X_valid.shape)

      # Confusion Matrix format
      # TN FP
      # FN TP
      print(model_scores['confusion_matrix'])
      print("*" * 100)

    # calc model feature importance
    if calc_feature_importance:
      feature_names = X_train.columns.values.tolist()
      gain_importance_df = get_feat_importance_df(
          model, feature_names=feature_names)
      gain_importance_tables.append(gain_importance_df)
      
  return(scores, gain_importance_tables, models)


def display_scores(scores):
    ''' 
      Given a single value, just print it.
      Given an array of scores (generic, usually cross validated performance metrics), 
      return the average.

        
    '''

    if type(scores) is list:
      print("Scores: {0}\nMean: {1:.3f}".format(scores, np.mean(scores)))
    else:
      print("Score: {}".format(scores))



def performance_report(scores):
  ''' Helper function to view results from the output of
      manual_cross_validation.

      Note: This function can only plot scores, no

      Arguments:
        scores (dict <string>:list<float>): Dictionary containing the model 
        performance scores returned by manual cross validation.
            
  '''
  print("Validation Scores")
  print("-"*30)
  if type(scores) is dict or type(scores) is defaultdict:
    for name in scores.keys():
        print(name)
        display_scores(scores[name])
        print()

def get_feat_importance_df(model, feature_names=None):
  '''
    extract feature importance dataframe from xgboost model
    
  '''
  importance_df = pd.DataFrame(model.feature_importances_,
                                index=feature_names,
                                columns=['importance']).sort_values('importance')
  return importance_df


# helper to compute cross validated variable importance
def calc_mean_importance(importance_df_list):
  vars_dict = defaultdict(list)
  
  # add all the scores for each variable to a dictionary
  for df in importance_df_list:
    for row in df.iterrows():
      vars_dict[row[0]].append(row[1].values[0])
  
  for k, v in vars_dict.items():
    vars_dict[k] = np.round(np.mean(v), 4)
  
  importance_list = pd.Series(vars_dict)
  return importance_list.sort_values()


def plot_importance(importance_df, title='',
                    save_filepath=None, figsize=(10, 15)):
  fig, ax = plt.subplots(figsize=figsize)
  importance_df.plot.barh(ax=ax)
  if title:
    plt.title(title)
  plt.tight_layout()
  if save_filepath is None:
    plt.show()
  else:
    plt.savefig(save_filepath)
  plt.close()



def main():
  # Unit testing the functions here
  path = "../"
  df_train = pd.read_csv(os.path.join(path, 'data/train_data.csv'))
  df_train.index=pd.to_datetime(df_train.date_time)
  df_train.date_time = df_train.index

  df_test = pd.read_csv(os.path.join(path, 'data/test_data.csv'))
  df_test.index=pd.to_datetime(df_test.date_time)
  df_test.date_time = df_test.index


  # variable lists
  metadata = ['date', 'date_time', 'year', 'month', 'day', 'hour', 'season']
  indicator = [col for col in df_train.columns if col[0] == 'i']
  constant = [var for var in df_train.columns if len(df_train[var].value_counts()) == 1]
  codes = ['sp1', 'sp2', 'sp3', 'sp4', 'wwa', 'wa', 'w' ,'ww', 'pweather', 'weather']
  excluded = indicator + constant + codes + ['rgauge', 'sog', 'tabspeed', 'msl']
  vis_vars=['target_hr1', 'vis_hr1', 'fog_formation', 'fog_dissipation', 'transition']
  target = 'target_hr1'

  categorical=['fog_state', 'season', 'tsig1', 'tsig2', 'tsig3', 'pchar'] #'w', 'ww', 'pweather',
               #'weather']
  discrete = [var for var in df_train.columns if len(df_train[var].unique()) < 15 and 
               var not in excluded + categorical + metadata + codes + indicator + vis_vars]

  continuous = [var for var in df_train.columns if var not in discrete + excluded + categorical + metadata + codes + indicator + vis_vars]
  
  numerical = discrete+continuous
  dates = df_train.date_time
  X = df_train[numerical + categorical + vis_vars].reset_index(drop=True)
  y = X.pop(target)

  print("Train/valid:", df_train.shape)
  print("Test:", df_test.shape)
  vars_sel = ['vis', 'temp_dew_dist', 'rh', 'ceiling', 'duration', 'hsig2', 'dni', 
                  'dewpt', 'drybulb', 'cbl', 'hlc', 'ntot', 'speed', 'vp', 'pchar','dir']
  num_vars_sel = [var for var in vars_sel if var in numerical]
  cat_vars_sel = [var for var in vars_sel if var in categorical]
  # creating training sets using only the selected features
  X_train, X_test = preprocess(df_train, df_test, cat_vars=cat_vars_sel, num_vars=num_vars_sel, cat_encoder='oh')
  y_train = y.copy()
  y_test = df_test[target]

  # Model training and testing
  print("Fitting model...")
  model = XGBClassifier(objective='binary:logistic', random_state=seed)
  model.fit(X_train, y_train)

  print("Scoring model...")
  scores, final_plots = score_model(model, X_train, X_test, y_train, y_test, df_test)
  performance_report(scores)

  final_plots['pr_plot'].plot()
  plt.show()

if __name__ == '__main__':
  main()


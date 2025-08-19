import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, scale
from lassonet import LassoNetRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import kurtosis, skew
import openml
import torch
import torch.optim as optim
import random 
import time
from joblib import Parallel, delayed
from pytabkit import RealMLP_TD_Regressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from rtdl_revisiting_models import FTTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
#Functions
def delete_missing_values(df, threshold=0.7):
  missing_col_mask = pd.isnull(df).mean(axis=0) < threshold
  df = df.loc[:, missing_col_mask] #localize columns where the mask is True
  return df

def remove_low_cardinality(df, threshold=10):
  cardinality = df.nunique(axis=0) > threshold
  df = df.loc[:, cardinality]
  return df

def delete_row_missing_values(df):
  row_mask = pd.isnull(df).sum(axis=1) == 0
  df = df.loc[row_mask, :] #localize rows where the mask is True
  return df

def full_preprocessing(df): #only on X features
  target_col = df.columns[-1]
  X_df = df.drop(df.columns[-1], axis=1)
  X_preprocessed= delete_missing_values(X_df)
  X_preprocessed= remove_low_cardinality(X_preprocessed)
  X_preprocessed= delete_row_missing_values(X_preprocessed)
  df_preprocessed = X_preprocessed.copy()
  df_preprocessed[target_col] = df[target_col].reindex(X_preprocessed.index) #realign the target with the processed features
  return df_preprocessed

def get_split_type(model_name):
  if model_name == "RandomForest" or model_name == "XGBoost":
    return "standard"
  else:
    return "quantile"

def split_data(X, y, type_scale="standard", rng=None):
  if type_scale == "quantile":
    scaler = QuantileTransformer(output_distribution="normal", random_state=rng)
    
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=rng)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.3, random_state=rng)
  X_train, y_train = X_train[:10000], y_train[:10000]
  X_val, y_val = X_val[:50000], y_val[:50000] #Truncate validation and test to 50,000 samples as said in the benchmark paper
  X_test, y_test = X_test[:50000], y_test[:50000]
  if type_scale == "quantile":
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

  X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
  X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.float32)
  X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
  
  return X_train, y_train, X_val, y_val, X_test, y_test

def n_folds(X_test):
  if X_test.shape[0] > 6000:
      n_iter = 1
  elif X_test.shape[0] > 3000:
      n_iter = 2
  elif X_test.shape[0] > 1000:
      n_iter = 3
  else:
      n_iter = 5
      
  return n_iter

def check_heavy_tail(y):
  skewness = skew(y)
  kurt = kurtosis(y, fisher=True) #fisher common definition of Kurtosis
  heavy_tailed = (
        abs(skewness) > 0.5 and        # Skewness indicates right skew (possible heavy tail)
        kurt > 3             # High kurtosis suggests heavy-tailed
    )  
  return heavy_tailed

def generate_hyperparameters_space(model_name):
  parameters = None
  if model_name == "LassoNet":
    parameters = {
      "lambda_indexes" : np.arange(5, 400, 10),
      "num_layers" : range(1, 9),
      "layer_sizes" : range(16, 1025),
      "batch_sizes" : [256, 512, 1024],
      "M_list" : [10, 50, 100]
    }
  
  elif model_name == "MLP":
    parameters = {
      "num_layers" : range(1, 9),
      "layer_sizes" : range(16, 1025),
      "batch_sizes" : [256, 512, 1024]
    }
    
  elif model_name == "XGBoost":
    parameters = {
      "max_depths" : range(1, 12),
      "num_estimators" : range(100, 6001, 200)
    }

  elif model_name == "RandomForest":
    parameters = {
      "max_depths" : [None, 2, 3, 4],
      "max_depths_probabilities" : [0.7, 0.1, 0.1, 0.1],
      "criterions" : ['squared_error', 'absolute_error'],
      "max_features" : ['sqrt', 'sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      "min_samples_splits" : [2, 3],
      "min_samples_splits_probabilities" : [0.95, 0.05]
    }
  elif model_name == "RealMLP":
    parameters = {
      "num_layers" : range(1, 9),
      "layer_sizes" : range(16, 1025),
      "batch_sizes" : [256, 512, 1024]
    }
  
  elif model_name == "TabNet":
    parameters = {
      "num_steps" : range(3, 6),
      "n_ds" : [8, 16, 24],
      "n_as" : [8, 16, 24],
      "gammas" : [1.0, 1.2, 1.5, 2.0],
      "lambda_sparses" : [0, 0.0001, 0.001, 0.01, 0.1],
      "learning_rates" : [0.005, 0.01, 0.02, 0.025],
      "batch_sizes" : [64, 256]
    }
    
  elif model_name == "FT-Transformer":
    parameters = {
      "num_layers" : [1, 2, 3, 4, 5, 6],
      "residual_dropouts" : [0, 0.5],
      "attention_dropouts" : [0, 0.1, 0.2, 0.3, 0.4, 0.5],
      "batch_sizes" : [64, 256]
    }
  return parameters

def FT_train(model, train_loader, val_loader, test_loader, lr, n_epochs):
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  y_pred_val = 0
  y_pred_test = 0
  for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
      optimizer.zero_grad()
      preds = model(xb, None)
      loss = loss_fn(preds, yb)
      loss.backward()
      optimizer.step()
    
    # Free unused cached GPU memory after each epoch
    torch.cuda.empty_cache()

    if epoch == n_epochs - 1:
      model.eval()
      val_preds = []
      with torch.no_grad():
          for xb, yb in val_loader:
              preds = model(xb, None)
              val_preds.append(preds.cpu())
              
      test_preds = []
      with torch.no_grad():
          for xb, yb in test_loader:
              preds = model(xb, None)
              test_preds.append(preds.cpu())
              
      y_pred_val = torch.cat(val_preds).numpy()
      y_pred_test = torch.cat(test_preds).numpy()

  return y_pred_val, y_pred_test

def generate_config(model_name, n_iterations, n_shuffles):
  "The configs have been inspired by the hyperparameter grids provided in the tabular data benchmark paper"
  models = []
  shuffle_configs = []
  parameters = generate_hyperparameters_space(model_name)
  for _ in range(n_iterations-1):
    parameters_dict = {}
    if model_name == "LassoNet":
        lambda_index = random.choice(parameters["lambda_indexes"])
        M = random.choice(parameters["M_list"])
        lr = log_uniform_distribution(1e-4, 1e-2)
        n_layers = random.choice(parameters["num_layers"])
        layer_size = random.choice(parameters["layer_sizes"])
        batch_size = random.choice(parameters["batch_sizes"])
        hidden_dims = (layer_size,) * n_layers
        
        parameters_dict = { 
          "lambda_index" :  lambda_index,
          "M" :  M,
          "lr" :  lr,
          "batch_size" :  batch_size,
          "hidden_dims" :  hidden_dims,
        }
    elif model_name == "MLP":
      lr = log_uniform_distribution(1e-4, 1e-2)
      n_layers = random.choice(parameters["num_layers"])
      layer_size = random.choice(parameters["layer_sizes"])
      batch_size = random.choice(parameters["batch_sizes"])
      hidden_dims = (layer_size,) * n_layers
      
      parameters_dict = { 
          "lr" :  lr,
          "batch_size" :  batch_size,
          "hidden_dims" :  hidden_dims,
        }
    
    elif model_name == "XGBoost":
      max_depth = random.choice(parameters["max_depths"])
      n_estimators = log_uniform_distribution(100, 2000, isint=True)
      min_child_weight = log_uniform_distribution(1, 1e2, isint=True)
      lr = log_uniform_distribution(1e-5, 0.7)
      gamma = log_uniform_distribution(1e-8, 7)
      lambd = log_uniform_distribution(1, 4)
      alpha = log_uniform_distribution(1e-8, 1e2)
      
      parameters_dict = { 
          "max_depth" :  max_depth,
          "n_estimators" :  n_estimators,
          "min_child_weight" :  min_child_weight,
          "lr" :  lr,
          "gamma" :  gamma,
          "lambd" :  lambd,
          "alpha" :  alpha,
        }
      
    
    elif model_name == "RandomForest":
      max_depth = random.choices(parameters["max_depths"], weights=parameters["max_depths_probabilities"], k=1)[0]
      n_estimators = log_uniform_distribution(9.5, 1000.5, isint=True)
      criterion = random.choice(parameters["criterions"])
      max_feature = random.choice(parameters["max_features"])
      min_samples_split = random.choices(parameters["min_samples_splits"], weights=parameters["min_samples_splits_probabilities"], k=1)[0]

      parameters_dict = { 
          "max_depth" :  max_depth,
          "n_estimators" :  n_estimators,
          "criterion" :  criterion,
          "max_feature" :  max_feature,
          "min_samples_split" :  min_samples_split,
        }
    
    elif model_name == "RealMLP":
      lr = log_uniform_distribution(1e-4, 1e-2)
      n_layers = random.choice(parameters["num_layers"])
      layer_size = random.choice(parameters["layer_sizes"])
      batch_size = random.choice(parameters["batch_sizes"])
      hidden_dims = [layer_size,] * n_layers
      
      parameters_dict = { 
          "lr" :  lr,
          "batch_size" :  batch_size,
          "hidden_dims" :  hidden_dims,
        }
    
    elif model_name == "TabNet":
      lr = random.choice(parameters["learning_rates"])
      lambda_sparse = random.choice(parameters["lambda_sparses"])
      n_steps = random.choice(parameters["num_steps"])
      n_a = random.choice(parameters["n_as"])
      n_d = random.choice(parameters["n_ds"])
      batch_size_tabnet = random.choice(parameters["batch_sizes"])
      gamma = random.choice(parameters["gammas"])
      
      parameters_dict = { 
          "lambda_sparse" :  lambda_sparse,
          "n_steps" :  n_steps,
          "n_a" :  n_a,
          "lr" :  lr,
          "n_d" :  n_d,
          "batch_size_tabnet" :  batch_size_tabnet,
          "gamma" :  gamma,
        }

    elif model_name == "FT-Transformer":
      lr = log_uniform_distribution(1e-5, 1e-2)
      n_layers = random.choice(parameters["num_layers"])
      res_dropout = random.choice(parameters["residual_dropouts"])
      attention_dropout = random.choice(parameters["attention_dropouts"])
      batch_size_ft = random.choice(parameters["batch_sizes"])
      
      parameters_dict = { 
          "res_dropout" :  res_dropout,
          "n_layers" :  n_layers,
          "attention_dropout" :  attention_dropout,
          "lr" :  lr,
          "batch_size_ft" :  batch_size_ft,
        }
      
    else:
      print('Model not in the code yet.')
      return None
        
    models.append(parameters_dict)
    
  for _ in range(n_shuffles):
    empty_dict = {}
    configs = [empty_dict]
    random.shuffle(models)
    configs += models
    shuffle_configs.append(configs)
  
  return shuffle_configs


def log_uniform_distribution(low, high, size=1, isint=False):
    values = np.random.uniform(np.log10(low), np.log10(high), size=size)
    values = 10 ** values
    if isint:
        values = np.round(values).astype(int)
    return values if size > 1 else values[0]

#helper function to process each fold
def process_fold(i, X_train, y_train, X_val, y_val, X_test, y_test, model_name, model, unsupervised_model=None, 
                 lambda_index=None, model_parameters=None, batch_size_tabnet=None, batch_size_ft=8, 
                 device='cpu', n_layers=None, attention_dropout=0.2, res_dropout=0.0, lr=0.001):
    start = 0
    end = 0
    result = {
        'n_fold': int(i + 1),
        'val_score': None,
        'test_score': None,
        'fit_time': None,
        'parameters': model_parameters if model_parameters is not None else "No hyperparameters"
    }

    if model_name == "LassoNet":
      start = time.time()
      path = model.path(X_train, y_train, return_state_dicts=True)  
      lambda_index = lambda_index % len(path)
      selected = path[lambda_index].selected
      model.fit(X_train[:, selected], y_train, dense_only=True)
      end = time.time()
      y_pred_val = model.predict(X_val[:, selected])
      y_pred_test = model.predict(X_test[:, selected])

    elif model_name == "XGBoost":
      start = time.time()
      model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
      end = time.time()
      y_pred_val = model.predict(X_val)
      y_pred_test = model.predict(X_test)

    elif model_name == "TabNet":
      y_train1 = y_train.reshape(-1, 1)
      y_val1 = y_val.reshape(-1, 1)
      start = time.time()
      unsupervised_model.fit(
          X_train=X_train,
          eval_set=[X_val],
          pretraining_ratio=0.2,
          max_epochs=10,
          batch_size=batch_size_tabnet,
          num_workers=0,
          drop_last=True
      )
      model.fit(
          X_train=X_train, y_train=y_train1,
          eval_set=[(X_train, y_train1), (X_val, y_val1)],
          eval_name=['train', 'val'],
          eval_metric=['mse'],
          max_epochs=50,
          batch_size=batch_size_tabnet,
          from_unsupervised=unsupervised_model
      )
      end = time.time()
      y_pred_val = model.predict(X_val)
      y_pred_test = model.predict(X_test)

    elif model_name == "FT-Transformer":
      X_train = torch.tensor(X_train, dtype=torch.float32)
      y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) #expects target tensor to be 2D
      X_val = torch.tensor(X_val, dtype=torch.float32)
      y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
      X_test = torch.tensor(X_test, dtype=torch.float32)
      y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

      X_train, X_val, X_test, y_train, y_val, y_test = (
          X_train.to(device), X_val.to(device), X_test.to(device),
          y_train.to(device), y_val.to(device), y_test.to(device)
      )

      train_ds = TensorDataset(X_train, y_train)
      val_ds = TensorDataset(X_val, y_val)
      test_ds = TensorDataset(X_test, y_test)

      n_cont_features = X_train.shape[1]

      if not model_parameters:
        #had to initialize values for all these parameters or else the model was not running
          model = FTTransformer(
              n_cont_features=n_cont_features,
              cat_cardinalities=[],
              d_out=1,
              n_blocks=3,
              d_block=192,
              attention_n_heads=8,
              attention_dropout=0.2,
              ffn_d_hidden=None,
              ffn_d_hidden_multiplier=4/3,
              ffn_dropout=0.1,
              residual_dropout=0.0,
              linformer_kv_compression_ratio=0.2,
              linformer_kv_compression_sharing='headwise'
          )
      else:
        model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=[],
            d_out=1, #regression
            d_block=192, 
            n_blocks=n_layers,
            attention_n_heads=8,
            attention_dropout=attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4/3,
            ffn_dropout=0.1,
            residual_dropout=res_dropout,
            linformer_kv_compression_ratio=0.2,
            linformer_kv_compression_sharing='headwise'
        )
      model = model.to(device)
      train_loader = DataLoader(train_ds, batch_size=batch_size_ft, shuffle=True)
      val_loader = DataLoader(val_ds, batch_size=batch_size_ft)
      test_loader = DataLoader(test_ds, batch_size=batch_size_ft)
      start = time.time()
      y_pred_val, y_pred_test = FT_train(model, train_loader, val_loader, test_loader, lr, 50)
      end = time.time()
      y_val = y_val.cpu().numpy()
      y_test = y_test.cpu().numpy()
    else:
      start = time.time()
      model.fit(X_train, y_train)
      end = time.time()
      y_pred_val = model.predict(X_val)
      y_pred_test = model.predict(X_test)

    fit_time = end - start
    val_score = r2_score(y_val, y_pred_val)
    test_score = r2_score(y_test, y_pred_test)
    result['val_score'] = val_score
    result['test_score'] = test_score
    result['fit_time'] = fit_time

    return result
  
def train(n_iterations, model_name, splits):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  results_df = pd.DataFrame(columns=["n_shuffle", "n_fold", "random_search_iteration", "r2_val", "r2_test", "fit_time", "hyperparameters"]) 
  index_shuffle = 0
  batch_size_tabnet = 8
  index = 0
  print(model_name)
  n_shuffles = 3
  shuffles = generate_config(model_name, n_iterations, n_shuffles)
  for shuffle in shuffles:
    index_iterations = 0
    for model_parameters in shuffle:
      model = None
      unsupervised_model = None
      lambda_index = None  
      attention_dropout = None
      n_layers = None
      batch_size_ft = None
      res_dropout = None
      batch_size_tabnet = 8
      lr = None
      if not model_parameters: #check if default hyperparameters
        if model_name == "LassoNet":
          model = LassoNetRegressor(device=device, n_iters=(75, 50))  
          lambda_index = 0
        elif model_name == "MLP":
            model = MLPRegressor(early_stopping=True) 
        elif model_name == "XGBoost":
            model = XGBRegressor(early_stopping_rounds=10) 
        elif model_name == "RandomForest":
            model = RandomForestRegressor(n_jobs=-1)
        elif model_name == "RealMLP":
            model = RealMLP_TD_Regressor(use_early_stopping=True)
        elif model_name == "TabNet":
            unsupervised_model = TabNetPretrainer(device_name=device)  
            model = TabNetRegressor(device_name=device)
            batch_size_tabnet = 8
        elif model_name == "FT-Transformer":
          batch_size_ft = 8
          attention_dropout = 0.2
          n_layers = 0
          res_dropout = 0.0
          lr = 0.001
      else:
        if model_name == "LassoNet":
          lambda_index = model_parameters["lambda_index"]
          M = model_parameters["M"]
          lr = model_parameters["lr"]
          batch_size = model_parameters["batch_size"]
          hidden_dims = model_parameters["hidden_dims"]
          model = LassoNetRegressor(
              hidden_dims=hidden_dims,
              batch_size=batch_size,
              n_iters=(75, 50),
              M=M,
              device=device,
          )
          model.optim_path = lambda params: optim.Adam(params, lr=lr)
        elif model_name == "MLP":
          lr = model_parameters["lr"]
          batch_size = model_parameters["batch_size"]
          hidden_dims = model_parameters["hidden_dims"]
          
          model = MLPRegressor(hidden_layer_sizes=hidden_dims, batch_size=batch_size, learning_rate_init=lr, early_stopping=True)
        elif model_name == "XGBoost":
          max_depth = model_parameters["max_depth"]
          n_estimators = model_parameters["n_estimators"]
          min_child_weight = model_parameters["min_child_weight"]
          lr = model_parameters["lr"]
          gamma = model_parameters["gamma"]
          lambd = model_parameters["lambd"]
          alpha = model_parameters["alpha"]
          
          model = XGBRegressor(max_depth=max_depth, min_child_weight=min_child_weight, reg_lambda=lambd, gamma=gamma, 
                              alpha=alpha, n_estimators=n_estimators, early_stopping_rounds=10)
        elif model_name == "RandomForest":
          max_depth = model_parameters["max_depth"]
          n_estimators = model_parameters["n_estimators"]
          criterion = model_parameters["criterion"]
          max_feature = model_parameters["max_feature"]
          min_samples_split = model_parameters["min_samples_split"]

          model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_features=max_feature, max_depth=max_depth, 
                                        min_samples_split=min_samples_split, n_jobs=-1)
        elif model_name == "RealMLP":
          lr = model_parameters["lr"]
          batch_size = model_parameters["batch_size"]
          hidden_dims = model_parameters["hidden_dims"]
          
          model = RealMLP_TD_Regressor(hidden_sizes=hidden_dims, batch_size=batch_size, lr=lr, use_early_stopping=True)
        elif model_name == "TabNet":
          lr = model_parameters["lr"]
          lambda_sparse = model_parameters["lambda_sparse"]
          n_steps = model_parameters["n_steps"]
          n_a = model_parameters["n_a"]
          n_d = model_parameters["n_d"]
          batch_size_tabnet = model_parameters["batch_size_tabnet"]
          gamma = model_parameters["gamma"]
          
          unsupervised_model = TabNetPretrainer(
              optimizer_fn=torch.optim.Adam,
              optimizer_params=dict(lr=lr),
              mask_type='sparsemax',
              device_name=device,
              verbose=0,
              n_d=n_d, n_a=n_a, n_steps=n_steps,
              lambda_sparse=lambda_sparse,
              gamma=gamma
          )
          
          model = TabNetRegressor(
              optimizer_fn=torch.optim.Adam,
              optimizer_params=dict(lr=lr),
              mask_type='sparsemax', # This will be overwritten if using pretrain model
              device_name=device, 
              verbose=0
          )

        elif model_name == "FT-Transformer":
          lr = model_parameters["lr"]
          n_layers = model_parameters["n_layers"]
          res_dropout = model_parameters["res_dropout"]
          attention_dropout = model_parameters["attention_dropout"]
          batch_size_ft = model_parameters["batch_size_ft"] 
          
        else:
          print('Model not in the code yet.')
          return None
      
      #run parallel on folds  
      fold_results = Parallel(n_jobs=-1)(
          delayed(process_fold)(
              i, X_train, y_train, X_val, y_val, X_test, y_test, model_name, model, 
              unsupervised_model, lambda_index, model_parameters, batch_size_tabnet, 
              batch_size_ft=batch_size_ft, device=device, n_layers=n_layers, 
              attention_dropout=attention_dropout, res_dropout=res_dropout, lr=lr
          ) 
          for i, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(splits)
      )  
        
      for fold_result in fold_results:
        results_df.loc[index] = [
            int(index_shuffle + 1),
            fold_result['n_fold'],
            int(index_iterations + 1),
            fold_result['val_score'],
            fold_result['test_score'],
            fold_result['fit_time'],
            fold_result['parameters']
        ]
        index += 1
      results_df["n_fold"] = results_df["n_fold"].astype(int)
      results_df["random_search_iteration"] = results_df["random_search_iteration"].astype(int) #reassure that both types are int
      index_iterations += 1
    index_shuffle += 1
  
  return results_df

if __name__ == '__main__':
    #Numerical regression
  dataset_ids = {
    "cpu_act": 44132,
    "pol": 44133,
    "elevators": 44134,
    "wine_quality": 44136,
    "Ailerons": 44137,
    "yprop_4_1": 45032,
    "houses": 44138,
    "house_16H": 44139,
    "delays_zurich_transport": 45034,
    "diamonds": 44140,
    "Brazilian_houses": 44141,
    "Bike_Sharing_Demand": 44142,
    "nyc-taxi-green-dec-2016": 44143,
    "house_sales": 44144,
    "sulfur": 44145,
    "medical_charges": 44146,
    "MiamiHousing2016": 44147,
    "superconduct": 44148
  } #ids received from openml website 


  datasets = {}
  for name, dataset_id in dataset_ids.items():
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)  # Load dataset features and target
    df = pd.DataFrame(X, columns=attributes)
    df[dataset.default_target_attribute] = y
    datasets[name] = df
      
  random_state = 42    
  preprocessed_datasets = {name: full_preprocessing(df) for name, df in datasets.items()}
  standard_precomputed_splits = {}
  quantile_precomputed_splits = {}
  for name in preprocessed_datasets:
    df = preprocessed_datasets[name]
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    #no need to scale for tree-based models
    standard_splits = [split_data(X, y, type_scale="standard", rng=random_state+i) for i in range(n_iter)]
    standard_precomputed_splits[name] = standard_splits
    log_scale = check_heavy_tail(y)
    y = np.log(y + 1e-6) if log_scale else scale(y)  # Apply transformation once
    X_test = split_data(X, y, type_scale="quantile", rng=random_state)[4]  # Get X_test for n_folds
    n_iter = n_folds(X_test)
    quantile_splits = [split_data(X, y, type_scale="quantile", rng=random_state+i) for i in range(n_iter)]
    quantile_precomputed_splits[name] = quantile_splits

  n_iterations = 25

  model_names = ["LassoNet"]
  
  for model_name in model_names:
    big_df = []
    for name in preprocessed_datasets:
      df = preprocessed_datasets[name]
      split_type = get_split_type(model_name)
      splits = None
      if split_type == "standard":
        splits = standard_precomputed_splits[name]
      else:
        splits = quantile_precomputed_splits[name]
      result_df =  train(n_iterations, model_name, splits)
      result_df["dataset_name"] = name
      result_df["model_name"] = model_name
      big_df.append(result_df)
    final_df = pd.concat(big_df, ignore_index=True)
    final_df.to_pickle(f"results_{n_iterations}iter_{model_name}1.pkl")
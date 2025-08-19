import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, scale
from lassonet import LassoNetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import kurtosis, skew
import openml
import torch
import torch.optim as optim
from pytabkit import RealMLP_TD_Regressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from rtdl_revisiting_models import FTTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from shap.explainers import GradientExplainer
from sklearn.datasets import make_friedman1

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
  df_preprocessed[target_col] = df[target_col].reindex(X_preprocessed.index)
  return df_preprocessed

def split_data(X, y, type_scale="standard", rng=None):
  if type_scale == "standard":
    scaler = StandardScaler()
  elif type_scale == "quantile":
    scaler = QuantileTransformer(output_distribution="normal", random_state=rng)
    
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=rng)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.3, random_state=rng)
  X_train, y_train = X_train[:10000], y_train[:10000]
  X_val, y_val = X_val[:50000], y_val[:50000] #Truncate validation and test to 50,000 samples as said in the paper
  X_test, y_test = X_test[:50000], y_test[:50000]

  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)

  X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
  X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.float32)
  X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
  
  return X_train, y_train, X_val, y_val, X_test, y_test

def FT_train(model, train_loader, val_loader, test_loader, lr, n_epochs):
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  y_pred_train = 0
  y_pred_val = 0
  y_pred_test = 0
  train_preds = []
  for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb, None)
        if epoch == n_epochs - 1:
          train_preds.append(preds.cpu())
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

    # Evaluate R² on validation set
    if epoch == n_epochs - 1:
      model.eval()
      val_preds = []
      with torch.no_grad():
          for xb, yb in val_loader:
              preds = model(xb, None)
              val_preds.append(preds.cpu())
      #test set         
      model.eval()
      test_preds = []
      with torch.no_grad():
          for xb, yb in test_loader:
              preds = model(xb, None)
              test_preds.append(preds.cpu())
      y_pred_train = torch.cat(train_preds).detach().numpy()        
      y_pred_val = torch.cat(val_preds).numpy()
      y_pred_test = torch.cat(test_preds).numpy()

  return y_pred_train, y_pred_val, y_pred_test

class TabNetWrapper(torch.nn.Module):
    __slots__ = ['tabnet_regressor'] #save memory
    
    def __init__(self, tabnet_regressor):
        super().__init__()
        self.tabnet = tabnet_regressor.network  # Use the underlying TabNet module
    def forward(self, x):
        return self.tabnet(x)[0]  # Return only predictions
      
# Define the wrapper for FT-Transformer
class FTTransformerWrapper(nn.Module):
    def __init__(self, ft_model):
        super().__init__()
        self.ft_model = ft_model

    def forward(self, x):
        out = self.ft_model(x, x_cat=None)
        if out.dim() == 1:
            out = out.unsqueeze(-1)  # make shape: (batch_size, 1)
        return out

def check_heavy_tail(y):
  skewness = skew(y)
  kurt = kurtosis(y, fisher=True) #fisher common definition of Kurtosis
  heavy_tailed = (
        abs(skewness) > 0.5 and        # Skewness indicates right skew (possible heavy tail)
        kurt > 3             # High kurtosis suggests heavy-tailed
    )  
  return heavy_tailed

def get_split_type(model_name):
  if model_name == "RandomForest" or model_name == "XGBoost":
    return "standard"
  else:
    return "quantile"
  

def shap_plot(model_name, model_parameters, df, name, column_names):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  batch_size_tabnet = 8
 
  model = None
  unsupervised_model = None
  X = df.drop(df.columns[-1], axis=1)
  y = df[df.columns[-1]]

  if name != 'freedman':
    log_scale = check_heavy_tail(y)
    y = np.log(y + 1e-6) if log_scale else scale(y)  # Apply transformation once
    split_type = get_split_type(model_name)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, type_scale=split_type, rng=42)
  
  else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.3, random_state=42)
    
  if name == 'freedman':
    model_parameters = "No hyperparameters"
  
  if model_parameters == "No hyperparameters": #check if default hyperparameters
    if model_name == "LassoNet":
      model = LassoNetRegressor(device=device, n_iters=(75, 50))  
      lambda_index = 0
    elif model_name == "MLP":
        model = MLPRegressor(early_stopping=True) 
    elif model_name == "XGBoost":
        model = XGBRegressor(early_stopping_rounds=50) 
    elif model_name == "RandomForest":
        model = RandomForestRegressor(n_jobs=-1)
    elif model_name == "RealMLP":
        model = RealMLP_TD_Regressor(use_early_stopping=True)
    elif model_name == "TabNet":
        unsupervised_model = TabNetPretrainer(device_name=device)  
        model = TabNetRegressor(device_name=device)
        batch_size_tabnet = 8
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
          n_iters=(25, 25),
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
                          alpha=alpha, n_estimators=n_estimators)
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
    
  if model_name == "LassoNet":
    path = model.path(X_train, y_train, return_state_dicts=True)  
    
    while lambda_index >= len(path):
      lambda_index -= len(path)
    
    selected = path[lambda_index].selected
    model.fit(X_train[:, selected], y_train, dense_only=True)
    
  elif model_name == "XGBoost":
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])
    shap_data = pd.DataFrame(X_test[:100], columns=column_names)  
  elif model_name == "TabNet":
    y_train1 = y_train.reshape(-1, 1)
    y_val1 = y_val.reshape(-1, 1)
    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_val],
        pretraining_ratio=0.2,
        max_epochs=5,
        batch_size=batch_size_tabnet, 
        num_workers=0, 
        drop_last=False  # Ensure all data is used
    )
    
    model.fit(
        X_train=X_train, y_train=y_train1,
        eval_set=[(X_train, y_train1), (X_val, y_val1)],
        eval_name=['train', 'val'],
        eval_metric=['mse'],
        max_epochs=100,
        batch_size=batch_size_tabnet,
        from_unsupervised=unsupervised_model
    )
    
    tabnet_wrapped = TabNetWrapper(model)
    explainer = GradientExplainer(tabnet_wrapped, torch.tensor(X_train[:100], dtype=torch.float32).to('cuda:0'))
    shap_values = explainer.shap_values(torch.tensor(X_test[:100], dtype=torch.float32).to('cuda:0'))
    shap_data = pd.DataFrame(X_test[:100], columns=column_names)
  
  elif model_name == "FT-Transformer":
    if freedman:
      X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
      y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1) #FT-Transformer expects 2-dimensional output

      X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
      y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)

      X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
      y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)
    else:
      X_train = torch.tensor(X_train, dtype=torch.float32)
      y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) #FT-Transformer expects 2-dimensional output

      X_val = torch.tensor(X_val, dtype=torch.float32)
      y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

      X_test = torch.tensor(X_test, dtype=torch.float32)
      y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    X_train, X_val, X_test, y_train, y_val, y_test = X_train.to(device), X_val.to(device), X_test.to(device), y_train.to(device), y_val.to(device), y_test.to(device)

    # Create TensorDatasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    n_cont_features = X_train.shape[1]
    
    if model_parameters == "No hyperparameters":
      batch_size_ft = 8 #random batch_size for default model=
      lr = 0.001  #default learning rate
      model = FTTransformer(
          n_cont_features=n_cont_features,
          cat_cardinalities=[],
          d_out=1,                      # Single regression output
          n_blocks=3,
          d_block=192,
          attention_n_heads=8,
          attention_dropout=0.2,
          ffn_d_hidden=None,
          ffn_d_hidden_multiplier=4 / 3,
          ffn_dropout=0.1,
          residual_dropout=0.0,
          linformer_kv_compression_ratio=0.2,           # values taken from https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md
          linformer_kv_compression_sharing='headwise',  # <---
      )
    else:
      model = FTTransformer(
          n_cont_features=X_train.shape[1], 
          cat_cardinalities=[],
          d_out=1,
          d_block=192,
          n_blocks=n_layers,
          attention_n_heads=8,
          attention_dropout=attention_dropout,
          ffn_d_hidden=None,
          ffn_d_hidden_multiplier=4 / 3,
          ffn_dropout=0.1,
          residual_dropout=res_dropout,
          linformer_kv_compression_ratio=0.2,           # values taken from https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md
          linformer_kv_compression_sharing='headwise',  # <---
      )
    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size_ft, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_ft)
    test_loader = DataLoader(test_ds, batch_size=batch_size_ft)
    y_pred_train, _, _ = FT_train(model, train_loader, val_loader, test_loader, lr, 50)
    wrapped_ft_model = FTTransformerWrapper(model).to(device)
    wrapped_ft_model.eval() #has to be used for SHAP
    X_train_tensor = X_train[:100].clone().detach().requires_grad_(True).to(device)
    X_test_tensor = X_test[:100].clone().detach().to(device)
    explainer = shap.GradientExplainer(wrapped_ft_model, X_train_tensor)
    shap_values = explainer.shap_values(X_test_tensor)
    X_test_np = X_test_tensor.detach().cpu().numpy()
    shap_data = pd.DataFrame(X_test_np, columns=column_names)
  
  elif model_name == "RealMLP":
    X_train_df = pd.DataFrame(X_train, columns=column_names)  
    X_test_df = pd.DataFrame(X_test, columns=column_names)
    model.fit(X_train_df, y_train)
    def predict_with_dataframe(X):
      if isinstance(X, np.ndarray):
          X = pd.DataFrame(X, columns=column_names) #to fix the column names problem with RealMLP features in fit not the same as feature in model.predict
      return model.predict(X)
    
    explainer = shap.KernelExplainer(predict_with_dataframe, X_train_df.iloc[:100])
    shap_values = explainer.shap_values(X_test_df.iloc[:100], nsamples=100)
    shap_data = X_test_df.iloc[:100].copy()
  else:
    model.fit(X_train, y_train)
    if model_name == "RandomForest":
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(X_test[:100])
      shap_data = pd.DataFrame(X_test[:100], columns=column_names) #lighter SHAP computation
    else:
      explainer = shap.KernelExplainer(model.predict, X_train[:100])
      shap_values = explainer.shap_values(X_test[:100], nsamples=100)
      shap_data = pd.DataFrame(X_test[:100], columns=column_names)
    
  if model_name == "FT-Transformer" or model_name == "TabNet":
    shap_values = shap_values[:, :, 0] #since shape of shap_values in this case is (100, 100, 1)
    if model_name == "FT-Transformer":
      base_pred = float(np.mean(y_pred_train))
    else:
      base_pred = float(np.mean(model.predict(X_train[:100])))
    base_value_scalar = base_pred
  elif model_name != "LassoNet":
    base_value_scalar = float(np.mean(explainer.expected_value))
    
  base_values_array = np.array([base_value_scalar], dtype=np.float32)
  explanation = shap.Explanation(
      values=shap_values,
      base_values=base_values_array,
      data=shap_data,
      feature_names=column_names
  )
  
  shap.summary_plot(shap_values, shap_data, max_display=10,show=False, rng=42)
  plt.savefig(f'shap_plots/{name}shap_summary_{model_name}100.pdf',dpi=700) 
  plt.close()
  shap.plots.bar(explanation, show=False)
  plt.savefig(f'shap_plots/{name}shap_barplot_{model_name}100.pdf',dpi=700) 
  plt.close()

  
if __name__ == '__main__':
  freedman = True
  if not freedman:
    dataset_ids = {
      "Brazilian_houses": 44141,
      "house_16H": 44139,
      "nyc-taxi-green-dec-2016": 44143,
    }
    
    datasets = {}
    for name, dataset_id in dataset_ids.items():
      dataset = openml.datasets.get_dataset(dataset_id)
      X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)  # Load dataset features and target
      df = pd.DataFrame(X, columns=attributes)
      df[dataset.default_target_attribute] = y
      datasets[name] = df
      print(f"Downloaded dataset '{name}' with {X.shape[0]} samples and {X.shape[1]} features.")
        
    random_state = 42    
    preprocessed_datasets = {name: full_preprocessing(df) for name, df in datasets.items()}

  df_ft = pd.read_pickle("Pickle_datasets/results_50iter_FT-Transformer.pkl")
  df_mlp = pd.read_pickle("Pickle_datasets/results_50iter_MLP.pkl")
  df_xgb = pd.read_pickle("Pickle_datasets/results_50iter_XGBoost.pkl")
  df_rmlp = pd.read_pickle("Pickle_datasets/results_50iter_RealMLP.pkl")
  results_df = pd.concat([df_rmlp])
  model_names = results_df['model_name'].unique()
  model_dict = {name:idx for idx, name in enumerate(model_names)}
  number_of_models = len(model_names)
  feature_names = {}
  for model_name in model_names:
    print(model_name)
    subset_df = results_df.loc[results_df['model_name'] == model_name]
    best_param = subset_df.loc[subset_df['r2_val'].idxmax()]
    hyperparameters = best_param['hyperparameters']
    if not freedman:
      for name, _ in datasets.items():
        print(name)
        df = preprocessed_datasets[name]
        feature_names[name] = list(df.columns[:-1])
        column_names = feature_names[name]
        shap_plot(model_name, hyperparameters, df, name, column_names)
    else:
      X, y = make_friedman1(n_samples=1000, random_state=42, n_features=100)
      df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
      df['target'] = y
      column_names = list(df.columns[:-1])
      nname = 'freedman'
      shap_plot(model_name, hyperparameters, df, nname, column_names)
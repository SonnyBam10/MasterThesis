from lassonet import LassoNetRegressor
import torch
import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler, scale
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

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

def check_heavy_tail(y):
  skewness = skew(y)
  kurt = kurtosis(y, fisher=True) #fisher common definition of Kurtosis
  heavy_tailed = (
        abs(skewness) > 0.5 and        # Skewness indicates right skew (possible heavy tail)
        kurt > 3             # High kurtosis suggests heavy-tailed
    )  
  return heavy_tailed

def lasso_test(df, lambda_index):
  X = df.drop(df.columns[-1], axis=1)
  y = df[df.columns[-1]]
  log_scale = check_heavy_tail(y)
  y = np.log(y + 1e-6) if log_scale else scale(y) 
  X_train, y_train, X_val, _, _, _ = split_data(X, y, type_scale="quantile", rng=42)
  model = LassoNetRegressor(
          batch_size=32,
          n_iters=(75, 50),
          M=10,
          device=device,
        )
  model.optim_path = lambda params: torch.optim.Adam(params, lr=1e-3)
  path = model.path(X_train, y_train, return_state_dicts=True)  
  while lambda_index >= len(path):
    lambda_index -= len(path)

  selected = path[lambda_index].selected
  model.fit(X_train[:, selected], y_train, dense_only=True)

  y_pred_val = model.predict(X_val[:, selected])
 
  return model, path, y_pred_val

device = "cuda" if torch.cuda.is_available() else "cpu"
lambda_indexes = np.arange(5, 400, 5)
dataset_ids = {
    "cpu_act": 44132,
  }


datasets = {}
for name, dataset_id in dataset_ids.items():
  dataset = openml.datasets.get_dataset(dataset_id)
  X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)  # Load dataset features and target
  df = pd.DataFrame(X, columns=attributes)
  df[dataset.default_target_attribute] = y
  datasets[name] = df
  print(f"Downloaded dataset '{name}' with {X.shape[0]} samples and {X.shape[1]} features.")
  

for name, _ in dataset_ids.items():
  val_r2_values = []
  best_score = -np.inf
  model_params = []
  best_id = 0
  for lambda_index in lambda_indexes:
    df = datasets[name]
    model, path, val_score = lasso_test(df, lambda_index) 
    val_r2_values.append(val_score)  
    if val_score > best_score:
      best_score = val_score
      model_params = [model, path]
      best_id = lambda_index
  
  model, path = model_params
  plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
  plt.xlabel("Feature Index")
  plt.ylabel("Lambda at Removal (Feature Importance)")
  plt.title("Feature Importances for best_id Model")
  plt.savefig(f'FeatureImportancesfordataset{name}.pdf')
  plt.clf()
    
  plt.plot(lambda_index, val_r2_values, label='validation')
  plt.xlabel('Lambda (log scale)')
  plt.ylabel('R2-score')
  plt.legend()
  plt.savefig(f'LambdaR2scoredataset{name}.pdf')
  

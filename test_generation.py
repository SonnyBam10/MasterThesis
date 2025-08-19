import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  print(f"{model_name} Performance: R²={r2:.3f}")
  return r2
  
def filter_synthetic_data(X_train_scaled, X_gen_scaled, y_gen, threshold_percentage=90):
  nn = NearestNeighbors(n_neighbors=1)
  nn.fit(X_train_scaled)
  
  distances, _ = nn.kneighbors(X_gen_scaled)
  mean_distances = distances.mean(axis=1)
  threshold = np.percentile(distances, threshold_percentage)
  mask = mean_distances <= threshold
  filtered_X_gen = X_gen_scaled[mask]
  filtered_y_gen = y_gen[mask]
  
  return filtered_X_gen, filtered_y_gen

def check_synthetic_data(name, pickle_name):
  print(name)
  df = datasets[name]
  generated_df = pd.read_pickle(pickle_name)
  X = df.drop(df.columns[-1], axis=1)
  y = df[df.columns[-1]]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  X_gen = generated_df.drop(generated_df.columns[-1], axis=1)
  y_gen = generated_df[generated_df.columns[-1]]

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  X_gen_scaled = scaler.transform(X_gen)

  y_train = np.array(y_train)
  y_test = np.array(y_test)
  y_gen = np.array(y_gen)

  mlp_real = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
  train_and_evaluate(mlp_real, X_train_scaled, y_train, X_test_scaled, y_test, "MLP Real only")

  #Augmented
  X_aug = np.vstack([X_train_scaled, X_gen_scaled])
  y_aug = np.hstack([y_train, y_gen])

  mlp_aug = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
  train_and_evaluate(mlp_aug, X_aug, y_aug, X_test_scaled, y_test, "MLP Real + Synthetic (Augmented)")
  
  #Filtered 90%
  filtered_X_gen, filtered_y_gen = filter_synthetic_data(X_train_scaled, X_gen_scaled, y_gen, 90)

  X_aug = np.vstack([X_train_scaled, filtered_X_gen])
  y_aug = np.hstack([y_train, filtered_y_gen])

  train_and_evaluate(mlp_aug, X_aug, y_aug, X_test_scaled, y_test, "MLP Real + Synthetic (Filtered)")
  
  #Filtered 75%
  filtered_X_gen, filtered_y_gen = filter_synthetic_data(X_train_scaled, X_gen_scaled, y_gen, 75)

  X_aug = np.vstack([X_train_scaled, filtered_X_gen])
  y_aug = np.hstack([y_train, filtered_y_gen])

  train_and_evaluate(mlp_aug, X_aug, y_aug, X_test_scaled, y_test, "MLP Real + Synthetic (Filtered)")
  
def t_sne(X_scaled, X_gen_scaled, name):
  tsne = TSNE(perplexity=50, max_iter=2000, random_state = 42)
  
  df_tsne_scaled = tsne.fit_transform(X_scaled) 
  x_original = df_tsne_scaled[:, 0]
  y_original = df_tsne_scaled[:, 1]
  
  df_tsne_scaled = tsne.fit_transform(X_gen_scaled) 
  x_artificial = df_tsne_scaled[:, 0]
  y_artificial = df_tsne_scaled[:, 1]

  fig = plt.figure(figsize=(12,10))
  ax1 = fig.add_subplot(111)
  
  ax1.scatter(x_original, y_original, label='Original')
  ax1.scatter(x_artificial, y_artificial, label='Artificial')
  plt.title(f'{name} Original versus Artificial t-SNE projection')
  plt.legend()
  plt.savefig(f'{name}t-SNE projection1.pdf')
  
def t_sne_comparison(name, pickle_name):
  print(name)
  df = datasets[name]
  generated_df = pd.read_pickle(pickle_name)
  X = df.drop(df.columns[-1], axis=1)

  X_gen = generated_df.drop(generated_df.columns[-1], axis=1)
  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  X_gen_scaled = scaler.transform(X_gen)
  
  t_sne(X_scaled, X_gen_scaled, name)

if __name__ == '__main__':
  dataset_ids = {
    "Bike_Sharing_Demand": 44142,
    "sulfur": 44145,
    "wine_quality": 44136,
    "Brazilian_houses": 44141,
    "elevators": 44134
  }
  datasets = {}
  for name, dataset_id in dataset_ids.items():
      dataset = openml.datasets.get_dataset(dataset_id)
      X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)
      df = pd.DataFrame(X, columns=attributes)
      df[dataset.default_target_attribute] = y
      datasets[name] = df
      print(f"Downloaded dataset '{name}' with {X.shape[0]} samples and {X.shape[1]} features.")

  check_synthetic_data('wine_quality', 'Pickle_datasets/synthetic_data_wine.pkl')
  check_synthetic_data('elevators', 'Pickle_datasets/synthetic_data_elevators.pkl')
  check_synthetic_data('Bike_Sharing_Demand', 'Pickle_datasets/synthetic_data_bike.pkl')
  check_synthetic_data('sulfur', 'Pickle_datasets/synthetic_data_sulfur.pkl')
  t_sne_comparison('elevators', 'Pickle_datasets/synthetic_data_elevators.pkl')
  t_sne_comparison('wine_quality', 'synthetic_data_wine_without_full_preprocess_100epochs.pkl')
  
  
  
  
  
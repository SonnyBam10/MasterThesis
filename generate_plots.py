import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gget_results(df):
  random_searches = df['random_search_iteration'].max()
  n_folds = df['n_fold'].max()
  best_mean = df[df['random_search_iteration'] == 1]['r2_val'].mean()
  val_scores_list = []
  val_scores_list.append(best_mean)
  for i in range(1, random_searches):
    actual_mean = df[df['random_search_iteration'] == i+1]['r2_val'].mean()
    if actual_mean > best_mean:
        best_mean = actual_mean
    val_scores_list.append(best_mean)
  test_scores = []
  for i in range(n_folds):
    filtered_df = df[df['n_fold'] == i+1]
    test_score = filtered_df.loc[filtered_df['r2_val'].idxmax(), 'r2_test']
    test_scores.append(test_score)
  total_time = df['fit_time'].sum()
  mean_test_score = np.mean(test_scores)
  return val_scores_list, mean_test_score, total_time
  
def get_results(df):
  by_shuffle_dfs = df.groupby('n_shuffle')
  results = by_shuffle_dfs.apply(gget_results)
  val_scores_lists = [tup[0] for tup in results]
  val_scores_lists = np.array(val_scores_lists, dtype=float)
  mean_test_scores = [tup[1] for tup in results]
  total_times = [tup[2] for tup in results]
  mean_val_scores_list = np.mean(val_scores_lists, axis=0).tolist()
  mean_of_mean_test_score = np.mean(mean_test_scores)
  mean_total_time = np.mean(total_times)
  return mean_val_scores_list, mean_of_mean_test_score, mean_total_time

def normalization_score(lower, upper, v):
  if upper == lower:
    return 1
  score = (v - lower) / (upper - lower)
  return min(max(score, 0), 1)

def get_best_mean(mean_score_vector):
  l = len(mean_score_vector)
  for i in range(l-1):
    if mean_score_vector[i+1] < mean_score_vector[i]:
      mean_score_vector[i+1] = mean_score_vector[i]
  return mean_score_vector 

model_dict = {
    "RandomForest" : 0, 
    "XGBoost" : 1, 
    "LassoNet" : 2, 
    "TabNet" : 3,
    "MLP" : 4, 
    "RealMLP" : 5, 
    "FT-Transformer" : 6
  }
results_df = pd.read_pickle('results_50iter_RealMLP.pkl')
model_names = [k for _, k in enumerate(model_dict)]
number_of_models = len(model_names)
dataset_names = results_df['dataset_name'].unique()
number_of_datasets = len(dataset_names)
n_iterations = results_df['random_search_iteration'].max()
datasets_dict = {name:idx for idx, name in enumerate(dataset_names)}
mean_test_scores = np.zeros((number_of_datasets, number_of_models))
val_scores_tab = np.zeros((number_of_datasets, number_of_datasets, n_iterations))

for name in dataset_names:
  print(name, flush=True)
  subset_df = results_df.loc[results_df['dataset_name'] == name]
  for model_name in model_names:
    print(f"Model: {model_name}")
    subsubset_df = subset_df.loc[subset_df['model_name'] == model_name]
    mean_vals, mean_test_score, total_time = get_results(subsubset_df)
    print(f'{model_name} has fit during {total_time} seconds.') 
    #include more shap plots
    val_scores_tab[datasets_dict[name]][model_dict[model_name]] = mean_vals      
    mean_test_scores[datasets_dict[name]][model_dict[model_name]] = mean_test_score
  
  mean_across_models = np.mean(val_scores_tab[datasets_dict[name]], axis=1) #across models since models are rows
  mean_across_models = [max(v, 0.0) for v in mean_across_models] #clip negative values
  
  #bounds
  test_errors = [1-value for value in mean_across_models]
  lower_bound = 1 - np.median(test_errors)
  upper_bound = max(mean_across_models)
  
  normalization_across_models_iterations = np.zeros((number_of_datasets, number_of_models, n_iterations))
  
  for i in range(n_iterations):
    r2_scores_across_models = val_scores_tab[datasets_dict[name], :, i]
    r2_scores_across_models = [max(0.0, v) for v in r2_scores_across_models]
    normalization_scores = [normalization_score(lower_bound, upper_bound, v) for v in r2_scores_across_models]
    normalization_across_models_iterations[datasets_dict[name], :, i] = normalization_scores      
  
  val_scores_tab = normalization_across_models_iterations


model_results = {}

for model_name in model_names:
  model_idx = model_dict[model_name]  
  model_scores = val_scores_tab[:, model_idx, :] 
  # Compute summary statistics
  mean_score = np.mean(model_scores, axis=0)  # Mean score across datasets
  mean_score = get_best_mean(mean_score)

  model_results[model_name] = {
      'mean_score': mean_score,
  }
    
mean_test_scores_ = np.mean(mean_test_scores, axis=0).tolist()

iterations = np.arange(0, n_iterations, 1)

plt.figure(figsize=(12, 7))
for model_name, stats in model_results.items():
  mean_scores = stats['mean_score']
  plt.plot(iterations, mean_scores, label=f"{model_name} ", linewidth=2)
  
plt.xlabel("Iterations")
plt.ylabel("Normalized R2-score")
plt.title("Model Performance Over Iterations")
plt.legend()
plt.grid(True)
plt.savefig('val_scores_10sdatasets22.pdf')
plt.clf()  

colors = ['red', 'blue', 'yellow', 'cyan', 'green', 'purple']
plt.figure(figsize=(10, 6))
plt.bar(model_names, mean_test_scores_, color=colors)
plt.title('Mean Test Scores depending on the model')
plt.xlabel('Model Names')
plt.ylabel('Mean Test Scores')
plt.savefig('test_scores_bar_10sdatasets22.pdf')
plt.clf() 
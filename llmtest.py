import pandas as pd
import openml
from be_great import GReaT
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    dataset_ids = {"pol": 44133}
    datasets = {}
    for name, dataset_id in dataset_ids.items():
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attributes)
        df[dataset.default_target_attribute] = y
        datasets[name] = df
        print(f"Downloaded dataset '{name}' with {X.shape[0]} samples and {X.shape[1]} features.")

    df = datasets['pol'][:100]

    # Initialize model - Note: you're initializing 'model' twice here.
    
    # This second initialization will be the one used.
    model = GReaT(llm='distilgpt2', batch_size=4, epochs=50,
                  fp16=True, dataloader_num_workers=2)

    print("Starting model training...")
    model.fit(df)
    print("Model training finished. Generating synthetic data...")
    synthetic_data = model.sample(n_samples=100)
    print("Synthetic data generated.")
    # You might want to do something with synthetic_data here, like print its head
    print(synthetic_data.head())


if __name__ == '__main__':
    main()
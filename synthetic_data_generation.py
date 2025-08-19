import pandas as pd
import openml
from be_great import GReaT
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #delete useless warnings
    

if __name__ == "__main__":
    dataset_ids = {"sulfur": 44145}
    datasets = {}
    for name, dataset_id in dataset_ids.items():
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attributes)
        df[dataset.default_target_attribute] = y
        datasets[name] = df

    df = datasets['sulfur']
    number_of_rows = len(df)
    number_of_features = len(df.columns)
    
    #from https://github.com/tabularis-ai/be_great/tree/main
    model = GReaT(
        llm='gpt2',
        batch_size=16,       
        epochs=100,        
        fp16=True  # Enable half-precision training for faster computation and lower memory usage 
    )

    synthetic_data = model.sample(
        n_samples=2*number_of_rows,
        max_length=8*number_of_features, #set through various differents tries
    )

    synthetic_data.to_pickle(f"Pickle_datasets/synthetic_data_sulfur.pkl")
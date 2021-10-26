import argparse
import json
import os
try:
    import pandas as pd
    import torch
except:
    os.system('pip install pandas')
    os.system('pip install torch==1.5.1')
    import pandas as pd
    import torch
import torch.optim as optim
import torch.utils.data
print(torch.__version__)
from train.model import LSTM

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    #model = LSTM(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])
    model = LSTM(model_info['num_classes'], model_info['input_size'], model_info['hidden_size'], model_info['num_layers'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array
    
def get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    df = pd.read_csv(os.path.join(training_dir, "car_data.csv"))

    df = df.drop(['Car_Name', 'Kms_Driven'], axis=1)
    
    input_cols = ['Year', 'Selling_Price', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
    categorical_cols = ['Year', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
    output_cols = ['Present_Price']
    
    dataframe1 = df.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    
    
    # Create PyTorch Tensors from Numpy arrays
    inputs = torch.from_numpy(inputs_array).float()
    targets = torch.from_numpy(targets_array).float()
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    val_percent, test_percent = 0.2, 0.1
    val_size = int(len(dataset) * val_percent)
    test_size = int(len(dataset) * test_percent)
    train_size = len(dataset) - val_size - test_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size)

    return test_loader

def test(input_data, model):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # TODO: Paste the train() method developed in the notebook here.
    input_data = input_data
    input_data = input_data.unsqueeze(-1)
    input_data = input_data.to(device)
    output = model.forward(input_data)
    
    return output


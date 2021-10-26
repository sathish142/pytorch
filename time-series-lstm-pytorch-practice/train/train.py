import argparse
import json
import os
try:
    import pandas as pd
except:
    os.system('pip install pandas')
    os.system('pip install torch==1.5.1')
    import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
print(torch.__version__)
from model import LSTM

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
    
def _get_train_data_loader(batch_size, training_dir):
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

    return train_loader


def train(model, train_loader, epochs, optimizer, loss_fn, device):
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
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            batch_X = batch_X.unsqueeze(-1)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            model.zero_grad()
            output=model.forward(batch_X)
            loss=loss_fn(output.squeeze(),batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        if epoch%10 == 0:
            print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--num_classes', type=int, default=1, metavar='N',
                        help='size of the num classes (default: 1)')
    parser.add_argument('--input_size', type=int, default=1, metavar='N',
                        help='size of the input (default: 1)')
    parser.add_argument('--hidden_size', type=int, default=2, metavar='N',
                        help='size of the hidden dimension (default: 2)')
    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='size of the num layers (default: 1)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    #model = LSTM(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)
    model = LSTM(args.num_classes, args.input_size, args.hidden_size, args.num_layers).to(device)

    print("Model loaded with num_classes {}, input_size {}, hidden_size {}, num_layers {}.".format(
        args.num_classes, args.input_size, args.hidden_size, args.num_layers
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'num_classes': args.num_classes,
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
        }
        torch.save(model_info, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
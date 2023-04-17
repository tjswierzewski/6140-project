import argparse
from math import floor
import random
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import dot, float64, matmul, nn, optim, tensor, sparse_coo_tensor, sparse_csr_tensor, stack
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from create_numpy_from_data import SongList, Song, swap_song_index_to_X
import numpy as np
from time import perf_counter
from statistics import mean
import matplotlib as mpl
import matplotlib.pyplot as plt

class User_Item_Encoder(nn.Module):
    def __init__(self, user_features, item_features, layers, item_layers = None) -> None:
        super(User_Item_Encoder, self).__init__()
        user_layers = layers
        if item_layers == None:
            item_layers = layers
        assert item_layers[-1] == user_layers[-1]
            

        user_encoder_layers = []
        user_encoder_layers.append(nn.Linear(user_features, user_layers[0], dtype=float64))
        
        for i in range(len(user_layers[0:-1])):
            user_encoder_layers.append(nn.LeakyReLU())
            user_encoder_layers.append(nn.Linear(user_layers[i], user_layers[i+1], dtype=float64))
        user_encoder_layers.append(nn.Sigmoid())
        
        item_encoder_layers = []
        item_encoder_layers.append(nn.Linear(item_features, item_layers[0], dtype=float64))
        
        for i in range(len(item_layers[0:-1])):
            item_encoder_layers.append(nn.LeakyReLU())
            item_encoder_layers.append(nn.Linear(item_layers[i], item_layers[i+1], dtype=float64))
        item_encoder_layers.append(nn.Sigmoid())

        self.user_encoder_functions = nn.Sequential(*user_encoder_layers)
        self.item_encoder_functions = nn.Sequential(*item_encoder_layers)

    def user_encoder(self, users):
        return self.user_encoder_functions(users)

    def item_encoder(self, items):
        return self.item_encoder_functions(items)

    def forward(self, users, items):
        users = self.user_encoder(users)
        items = self.item_encoder(items)

        return torch.matmul(users, torch.transpose(items, 0 ,1))
    
class TrainingDataset(Dataset):
    def __init__(self, dataset, available) -> None:
        super(TrainingDataset).__init__()
        self.dataset = dataset
        self.available = available
        self.transformed_dataset = swap_song_index_to_X(self.dataset)
        [playlist, song, rank] = sparse.find(self.transformed_dataset)
        self.song_data = sparse_coo_tensor(np.vstack([song,playlist]), tensor(rank), size=self.transformed_dataset.shape[::-1])

    def __getitem__(self, index):
        return self.song_data.select(1, self.available[index]), np.trim_zeros(self.dataset[self.available[index]].toarray()[0]) - 1, self.available[index]
        
    def __len__(self):
        return len(self.available)
    
    def shape(self):
        return self.song_data.shape

class DataCollator(object):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __call__(self, batch):
        users, items, key = list(zip(*batch))
        songs = np.unique(np.concatenate(items))
        return stack(users), self.dataset.song_data.index_select(0, tensor(songs)), (key, songs)

class CustomLossFunction:
    def __init__(self) -> None:
        self.__loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, product, key):
        return self.__loss(product, key)


def check_model(data, data_loader, model, loss_function, optimizer = None):
    epoch_start_time = perf_counter()
    batch_times = []
    epoch_loss = []
    if optimizer:
        model.train()
    else:
        model.eval()

    for users, items, (user_key, item_key) in data_loader:
        batch_start_time = perf_counter()
        products = model(users, items)
        answers = []
        for user in user_key:
            test = tensor([np.where(item_key == x)[0][0] for x in sparse.find(data.dataset[user])[2] -1 if x in item_key])
            row = torch.zeros(len(item_key), dtype=torch.double)
            row[test] = 1
            answers.append(row)
        answers = torch.stack(answers)
        loss = loss_function(products, answers)

        epoch_loss.append(loss.item() * len(users))

        if optimizer and loss_function:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_end_time = perf_counter()
        batch_time = batch_end_time-batch_start_time
        batch_times.append(batch_time)
        # print(f"Batch Time: {batch_time:0.4f} seconds")
    epoch_end_time = perf_counter()
    print(f"Epoch Time: {epoch_end_time - epoch_start_time:0.4f} seconds\nAverage Batch Time: {mean(batch_times):0.4f}")
    return sum(epoch_loss) / len(data)

def random_split(length, size=0.1):
    sequence = list(range(length))
    random.shuffle(sequence)
    split = floor(length * (1-size))
    first = sequence[:split]
    second = sequence[split:]
    return list(sorted(first)), list(sorted(second))

def main():
    
    # Script Argument Parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--songlist", help = "Songlist Pickle", required= True)
    parser.add_argument("-m", "--matrix", help = "Data Matrix", required= True)
    parser.add_argument("-S", "--seed", help = "Seed", type= int, default= None)

    args = parser.parse_args()

    # Import SongList
    songlist = SongList()
    songlist.load(args.songlist)

    # Import Matrix
    matrix = sparse.load_npz(args.matrix)

    # Data creation
    train, validate = random_split(matrix.shape[0], 0.25)
    batch_size = 4092
    train_data = TrainingDataset(matrix, train)
    train_collator = DataCollator(train_data)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_collator)

    validate_data = TrainingDataset(matrix, validate)
    validate_collator = DataCollator(validate_data)
    validate_data_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False, collate_fn=validate_collator)
    
    learning_rate = 0.001
    momentum = 0.9
    model = User_Item_Encoder(*train_data.shape(), [500, 100], [250,100])
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum= momentum)
    loss_function = CustomLossFunction()
    
    num_epochs = 50
    train_loss_per_epoch = []
    validate_loss_per_epoch = []
    min_validate_loss = np.inf
    for epoch in range(num_epochs):
        print(f"{epoch}: Training") 
        training_loss = check_model(train_data, train_data_loader, model, loss_function, optimizer)
        train_loss_per_epoch.append(training_loss)
        print(f"Loss = {train_loss_per_epoch[-1]}")
        print(f"{epoch}: Validate")
        validate_loss = check_model(validate_data, validate_data_loader, model, loss_function)
        if validate_loss < min_validate_loss:
            min_validate_loss = validate_loss
            torch.save(model.state_dict(), "best_model.mdl")
        validate_loss_per_epoch.append(validate_loss)
        print(f"Loss = {validate_loss_per_epoch[-1]}")
    fig, ax = plt.subplots()
    ax.plot(train_loss_per_epoch)
    ax.plot(validate_loss_per_epoch)
    plt.show()




if __name__ == "__main__":
    main()


import argparse
from scipy import sparse
import torch
from torch import dot, float64, matmul, nn, optim, tensor, sparse_coo_tensor, sparse_csr_tensor, stack
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from create_numpy_from_data import SongList, Song, swap_song_index_to_X
import numpy as np

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
        users = F.normalize(self.user_encoder(users))
        items = F.normalize(self.item_encoder(items))

        return users, items
    
class TrainingDataset(Dataset):
    def __init__(self, dataset) -> None:
        super(TrainingDataset).__init__()
        self.dataset = dataset
        self.transformed_dataset = swap_song_index_to_X(self.dataset)
        [playlist, song, rank] = sparse.find(self.transformed_dataset)
        self.song_data = sparse_coo_tensor((song, playlist), tensor(rank))

    def __getitem__(self, index):
        return self.song_data.select(1,index), np.trim_zeros(self.dataset[index].toarray()[0]) - 1, index
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def shape(self):
        return self.song_data.shape

class DataCollator(object):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __call__(self, batch):
        users, items, key = list(zip(*batch))
        songs = np.unique(np.concatenate(items))
        return stack(users), self.dataset.song_data.index_select(0, tensor(songs)), (key, songs)


def check_model(data, data_loader, model, optimizer, loss_function, train = False):
    epoch_loss = []
    if train:
        model.train()
    else:
        model.eval()

    for users, items, (user_key, item_key) in data_loader:
        playlist_embeds, song_embeds = model(users, items)
        answers = data.transformed_dataset[list(user_key)]
        answers[answers != 0] = 1
        values = []
        [playlists, songs, ones] = sparse.find(answers)
        for i, playlist in enumerate(playlists):
            values.append(dot(playlist_embeds[playlist], song_embeds[np.where(item_key == songs[i])[0][0]]))
        loss = loss_function(F.sigmoid(stack(values)), tensor(ones))

        epoch_loss.append(loss.item() * len(users))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return sum(epoch_loss) / len(data)

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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Data creation
    batch_size = 128
    data = TrainingDataset(matrix)
    collator = DataCollator(data)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    
    learning_rate = 0.001
    momentum = 0.9
    model = User_Item_Encoder(*data.shape(), [500,250,100])
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum= momentum)
    loss_function = nn.BCELoss()
    
    num_epochs = 25
    loss_per_epoch = []
    for epoch in range(num_epochs):
        training_loss = check_model(data, data_loader, model, optimizer, loss_function, train=True)
        loss_per_epoch.append(training_loss)
        print(loss_per_epoch[-1])



if __name__ == "__main__":
    main()


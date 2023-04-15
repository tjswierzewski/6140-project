from torch import matmul, nn, optim, functional as F

class User_Item_Encoder(nn.Module):
    def __init__(self, user_features, item_features, layers, item_layers = None) -> None:
        super(User_Item_Encoder, self).__init__()
        user_layers = layers
        if item_layers == None:
            item_layers = layers

        user_encoder_layers = []
        user_encoder_layers.append(nn.Linear(user_features, user_layers[0]))
        
        for i in range(len(user_layers[0:-1])):
            user_encoder_layers.append(nn.ReLU())
            user_encoder_layers.append(nn.Linear(user_layers[i], user_layers[i+1]))
        user_encoder_layers.append(nn.Sigmoid())
        
        item_encoder_layers = []
        item_encoder_layers.append(nn.Linear(item_features, item_layers[0]))
        
        for i in range(len(item_layers[0:-1])):
            item_encoder_layers.append(nn.ReLU())
            item_encoder_layers.append(nn.Linear(item_layers[i], item_layers[i+1]))
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

        return matmul(F.normalize(users), F.normalize(items))


def main():
    learning_rate = 0.001
    momentum = 0.9
    model = User_Item_Encoder(100,500, [128,64,32])
    optimizer = optim.SGD(model.parameters, lr = 0.001, momentum= momentum)
    loss_function = nn.BCELoss()
    

if __name__ == "__main__":
    main()


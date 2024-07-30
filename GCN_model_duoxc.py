import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,PointGNNConv

class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class mlp_h(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputdim = 300
        self.hiddendim = 64
        self.outputdim = 3
        self.dropout_rate = 0
        self.fc1 = nn.Sequential(nn.Linear(self.inputdim, self.hiddendim),
                                   nn.Dropout(self.dropout_rate),
                                   nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.hiddendim, self.outputdim),
                                   nn.Dropout(self.dropout_rate),
                                   nn.ReLU())
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class mlp_f(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputdim = 303
        self.hiddendim = 300
        self.outputdim = 300
        self.dropout_rate = 0
        self.fc1 = nn.Sequential(nn.Linear(self.inputdim, self.hiddendim),
                                 nn.Dropout(self.dropout_rate),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.hiddendim, self.outputdim),
                                 nn.Dropout(self.dropout_rate),
                                 nn.ReLU())
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class mlp_g(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputdim = 300
        self.hiddendim = 300
        self.outputdim = 300
        self.dropout_rate = 0
        self.fc1 = nn.Sequential(nn.Linear(self.inputdim, self.hiddendim),
                                 nn.Dropout(self.dropout_rate),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.hiddendim, self.outputdim),
                                 nn.Dropout(self.dropout_rate),
                                 nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PointGNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes,n_layers):
        super(PointGNNModel, self).__init__()
        self.embed = nn.Sequential(nn.Linear(16, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 300),
                                   nn.ReLU(),
                                   )
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                PointGNNConv(mlp_h(),mlp_f(),mlp_g())
            )
        self.fc1 = nn.Linear(300, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 300)
        self.out = nn.Linear(300, 2)
        self.relu = nn.ReLU()
        self.dropout_rate = 0.2
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, pos, edge_index):
        x = self.embed(x)
        for pgcnn in self.gnn_layers:
            x = torch.relu(pgcnn(x, pos, edge_index))
        fully1 = self.relu(self.fc1(x))
        fully1 = self.dropout(fully1)
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = self.relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict




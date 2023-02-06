import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from simulated_data import return_simulated_dataset
from sklearn.model_selection import GroupShuffleSplit
# set device: cup or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    # 418 genes in DH_line, include drop out in encoder
    def __init__(self, seq_length,features_num=418, hidden_size=128, num_layers=2, dropout=0):
        super(Encoder, self).__init__()
        self.seq_len = seq_length
        self.n_features = features_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            features_num,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,dropout=dropout
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        x = x.reshape((1, self.seq_len, self.n_features))
        encoder_outputs, (hidden, cell) = self.lstm(x)

        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(
        self, feature_num=418, hidden_size=128, output_size=512, num_layers=2
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            feature_num,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        decoder_output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(decoder_output)
        return prediction, (hidden, cell)


class LSTM_AE(nn.Module):

    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # X: tensor of shape (batch_size, seq_length, hidden_size)
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()

        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history
def main():
    #generate simulated  dataset
    simulated_dataset = return_simulated_dataset()
    print(simulated_dataset)
    #number of days
    time_step = 60
    embedding_size = 418
    num_hiddens = 128
    num_layers = 2
    dropout = 0.1
    encoder = Encoder(time_step, embedding_size, num_hiddens, num_layers, dropout)
    decoder = Decoder(embedding_size, num_hiddens, num_layers)
    net = LSTM_AE(encoder, decoder)

    def create_dataset(df):
        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        print(dataset) #? the last one is empty
        n_seq, seq_len, n_features = torch.stack(dataset).shape

        return dataset, seq_len, n_features

    train_df = simulated_dataset.sample(frac=0.8, random_state=200)
    test_df = train_df.drop(train_df.index)
    train_dataset, seq_len, n_features = create_dataset(train_df)
    test_dataset, seq_len, n_features = create_dataset(test_df)
    train_model(net, train_dataset, test_dataset, 2)

if __name__ == '__main__':
    main()

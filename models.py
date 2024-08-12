import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_out, lstm_layers, out_dim, dropout_lstm, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Defining a bidirectional lstm
        self.bilstm = nn.LSTM(input_size=embed_size, hidden_size=lstm_out, num_layers=lstm_layers, batch_first=True,
                              dropout=dropout_lstm, bidirectional=True)
        self.dense1 = nn.Linear(lstm_out * 2, 20)
        self.dense2 = nn.Linear(20, 10)
        self.output = nn.Linear(10, out_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, input_ids, input_lengths, targets=None):
        x = self.embedding(input_ids)
        x = self.drop1(x)
        packed_x = pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.bilstm(packed_x)
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        # We can do this because it's bidirectional
        out = output_unpacked[:, 0, :]
        out = self.leaky_relu(out)

        out = self.leaky_relu(self.drop2(self.dense1(out)))
        out = self.leaky_relu(self.drop2(self.dense2(out)))

        out = self.output(out)
        out = self.sigmoid(out)

        out = out.view(-1)
        if targets is not None:
            # Calculate loss too if targets are provided
            criterion = nn.BCELoss()
            loss = criterion(out, targets.float())
            return out, loss
        else:
            return out,

from torch import nn

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=16, output_size=1, num_layers=1, d_list=None):
        super().__init__()
        self.d_list = d_list
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn   1, 16, 1, 1
        self.forwardCalculation = nn.Linear(hidden_size*20, output_size)

    def forward(self, _x, d_list):
        self.d_list = d_list
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        x = nn.utils.rnn.pad_packed_sequence(x,batch_first=True)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)  [20,128,16]
        x = x.view(b, s * h)  # [128,16*20]
        x = self.forwardCalculation(x)  # [128*20,1]
        # x = x.view(s, b, -1)  # [20,128,1]
        return x


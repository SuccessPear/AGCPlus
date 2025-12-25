import torch
import torch.nn as nn


def _output_dim(cfg):
    """
    Equivalent to num_classes in CNN.
    If not provided, default to input_dim (regression).
    """
    return cfg.get("output_dim", cfg.get("input_dim"))


class FlexibleRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        rnn_type="lstm",
        dropout=0.0,
        bidirectional=False,
        head="linear",
        output_dim=None,
    ):
        super().__init__()

        rnn_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[rnn_type.lower()]

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        if head == "linear":
            self.head = nn.Linear(rnn_out_dim, output_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(rnn_out_dim, rnn_out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(rnn_out_dim, output_dim),
            )
        else:
            raise ValueError("head must be 'linear' or 'mlp'")

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, T, F)
        """
        out, _ = self.rnn(x)       # (B, T, H)
        h_last = out[:, -1]        # many-to-one
        return self.head(h_last)

def build_my_rnn(cfg):
    input_dim     = cfg.get("input_dim", 5)
    hidden_dim    = cfg.get("hidden_dim", 128)
    num_layers    = cfg.get("num_layers", 2)
    rnn_type      = cfg.get("rnn_type", "lstm")   # rnn | lstm | gru
    dropout       = cfg.get("dropout", 0.0)
    bidirectional = cfg.get("bidirectional", False)
    head          = cfg.get("head", "linear")

    return FlexibleRNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        bidirectional=bidirectional,
        head=head,
        output_dim=_output_dim(cfg),
    )

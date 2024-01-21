import abc
from typing import Optional, Tuple

import deepsysid.metrics
import numpy as np
import torch
import math
from deepsysid.models.switching.switchrnn import SwitchingLSTMBaseModel, SwitchingLSTMBaseModelConfig
from deepsysid.networks.switching import SwitchingBaseLSTM, SwitchingLSTMOutput, UnconstrainedSwitchingLSTM
import torch.nn as nn
from torch.nn import LSTM
from typing import Literal




if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    a = torch.randn(5, 5, requires_grad=True)
    b = gram_schmidt(a)
    c = b.sum()
    c.backward()
    print(b.matmul(b.t()))
    print(a.grad)

class ControllableReLiNetSVD(SwitchingBaseLSTM):
    def __init__(
        self,
        control_dim: int,
        state_dim: int,
        output_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
        ) -> None:
        super().__init__()

        if not (state_dim >= output_dim):
            raise ValueError(
                'state_dim must be larger or equal to output_dim, '
                f'but {state_dim = } < {output_dim}.'
            )

        self.control_dim = control_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.recurrent_dim = recurrent_dim

        self.lstm = LSTM(
            input_size=control_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        """self.T = nn.Parameter(
            torch.from_numpy(np.random.normal(0, 1, (state_dim, state_dim))).float(),
            requires_grad=True,
        )"""

        "code from Github User legendongary"
        "https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py"

        def gram_schmidt(vv):
            def projection(u, v):
                return (v * u).sum() / (u * u).sum() * u

            nk = vv.size(0)
            uu = torch.zeros_like(vv, device=vv.device)
            uu[:, 0] = vv[:, 0].clone()
            for k in range(1, nk):
                vk = vv[k].clone()
                uk = 0
                for j in range(0, k):
                    uj = uu[:, j].clone()
                    uk = uk + projection(uj, vk)
                uu[:, k] = vk - uk
            for k in range(nk):
                uk = uu[:, k].clone()
                uu[:, k] = uk / uk.norm()
            return uu

        self.gen_A = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim * state_dim, bias=True
        )
        self.gen_B = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim * control_dim, bias=True
        )
        self.C = nn.Linear(in_features=state_dim, out_features=output_dim, bias=False)

        self.gen_U = nn.Linear(
            in_features=self.recurrent_dim, out_features=self.state_dim * self.state_dim, bias=True
        )
        self.gen_V = nn.Linear(
            in_features=self.recurrent_dim, out_features=self.control_dim * self.control_dim, bias=True
        )

    "@abc.abstractmethod"
    def forward(
            self,
            control: torch.Tensor,
            previous_output: torch.Tensor,
            previous_state: Optional[torch.Tensor] = None,
            hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SwitchingLSTMOutput:
        """
        :control: (batch, time, control)
        :previous_output: (batch, output)
        :previous_state: (batch, state) or None
        :hx: hx = (h0, c0) or None
        :returns: SwitchingLSTMOutput
            with .outputs.shape = (batch, time, output)
                 .states.shape = (batch, time, state)
                 .system_matrices = (batch, time, state, state)
                 .control_matrices = (batch, time, state, control)
        """
        "Vanilla Code von ReLiNet"
        batch_size = control.shape[0]
        sequence_length = control.shape[1]

        x, (h0, c0) = self.lstm.forward(control, hx=hx)

        "x_c im originalen Format des hidden States: x[batch, t, :]"
        x_c = x

        x = torch.reshape(x, (batch_size * sequence_length, self.recurrent_dim))

        A = torch.reshape(
            self.gen_A.forward(x),
            (batch_size, sequence_length, self.state_dim, self.state_dim),
        )


        n = self.state_dim
        m = self.control_dim
        l = math.ceil(n/m)
        K_c = torch.zeros(n, l * m)
        B = torch.zeros(batch_size, sequence_length, self.state_dim, self.control_dim)

        "Erzeugen von B normal"
        B[:, :, :, :] = torch.reshape(
            self.gen_B.forward(x),
            (batch_size, sequence_length, self.state_dim, self.control_dim),
        )

        "Theta erzeugen"
        theta = torch.zeros([batch_size, n, n], device=control.device)


        "U und V generieren"
        "init"
        U = torch.zeros([batch_size, n, n], device=control.device)
        V = torch.zeros([batch_size, n, l*m], device=control.device)

        U = torch.reshape(
            self.gen_U.forward(x),
            (batch_size, self.state_dim, self.state_dim),
        )

        V = torch.reshape(
            self.gen_V.forward(x),
            (batch_size, self.state_dim, self.control_dim * l),
        )

        "Gram Schmidt für jeweils U und V"

        "U * Theta * V:"

        "temp_K_c = torch.matmul(U, Theta)"

        "K_c = torch.matmul(temp_K_c, V)"

        for batch in batch_size:
            for t in range(l):
                "B[batch, t, :, :] = K_c[:][t*m:(t+1)*m-1]"
                B[batch, t, :, :] = torch.cat(torch.split(K_c.unsqueeze(0), 3, dim=2))[t, :, :]



        """
        method 3: construction of K_c via multiplication

        for batch in batch_size:
            K_c[:, :n] = torch.eye(n)
            for t in range(l):
                "B[batch, t, :, :] = K_c[:][t*m:(t+1)*m-1]"
                B[batch, t, :, :] = torch.cat(torch.split(K_c.unsqueeze(0), 3, dim=2))[t, :, :]

        B[:, l+1:, :, :] = torch.reshape(
            self.gen_B.forward(x),
            (batch_size, sequence_length-l, self.state_dim, self.control_dim),
        )


        """

        """
        Erzeugen der Input Matrizen, neue Version (Einsen nur auf der Diagonale)
        B = np.zeros(n,l*m)
        for i in range(l):
            B[i][i] = 1


        def B_i(i):
            if (i>l):
            raise ValueError(
                f'Only l matrices are produced for controllability, but i is greater than l'
            return B[:][i*(m-1)+1:i*m]
        """

        """
        "Version wie in Proposal beschrieben"
        def B_Controllable(i):
           "first zeros part"
            if(i>1):
                if(i<l):
                    np.zeros((i*m,m))
                else:
                    np.zeros((n-m,m))
           "Identity part"
           np.identity(n)

            "last zeros part"
            if (i < l):
                np.zeros((n-l+m, m))
        """

        """
        Pseudo Code Methode 3(2 Matrizen full rank):
        """

        states = torch.zeros(
            size=(batch_size, sequence_length, self.state_dim), device=control.device
        )
        if previous_state is None:
            state = torch.zeros(
                size=(batch_size, self.state_dim), device=control.device
            )
            state[:, : self.output_dim] = previous_output
        else:
            state = previous_state

        for time in range(sequence_length):
            state = (
                    A[:, time] @ state.unsqueeze(-1)
                    + B[:, time] @ control[:, time].unsqueeze(-1)
            ).squeeze(-1)
            state = state
            states[:, time] = state

        outputs = self.C.forward(states)

        return SwitchingLSTMOutput(
            outputs=outputs,
            states=states,
            hx=(h0, c0),
            system_matrices=A,
            control_matrices=B,
        )

    @property
    def output_matrix(self) -> torch.Tensor:
        """
        :returns: .shape = (output, state)
        """
        return self.C.weight

    @property
    def control_dimension(self) -> int:
        return self.control_dim

    @property
    def state_dimension(self) -> int:
        return self.state_dim

    @property
    def output_dimension(self) -> int:
        return self.output_dim



class ControllableReLiNetModelConfig(SwitchingLSTMBaseModelConfig):
    "SwitchingLSTMBaseModelConfig"
    recurrent_dim : int
    num_recurrent_layers: int
    dropout: float
    sequence_length:int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    loss = Literal['mse', 'msge']


class ControllableReLiNetSVDModel(SwitchingLSTMBaseModel):
    CONFIG = ControllableReLiNetModelConfig

    def __init__(self, config: ControllableReLiNetModelConfig) -> None:
        if config.switched_system_state_dim is None:
            state_dim = len(config.state_names)
        else:
            state_dim = config.switched_system_state_dim

        predictor = ControllableReLiNetSVD(
            control_dim=len(config.control_names),
            state_dim=state_dim,
            output_dim=len(config.state_names),
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            dropout=config.dropout,
        )
        super().__init__(config=config, predictor=predictor)
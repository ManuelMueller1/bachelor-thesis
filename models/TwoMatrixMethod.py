import abc
from typing import Optional, Tuple

import numpy as np
import torch
import math
from deepsysid.models.switching.switchrnn import SwitchingLSTMBaseModel, SwitchingLSTMBaseModelConfig
from deepsysid.networks.switching import SwitchingBaseLSTM, SwitchingLSTMOutput, UnconstrainedSwitchingLSTM
import torch.nn as nn
from torch.nn import LSTM

class ControllableReLiNet2MM(SwitchingBaseLSTM):
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

        self.gen_A = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim, bias=True
        )
        self.gen_B = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim * control_dim, bias=True
        )
        self.C = nn.Linear(in_features=state_dim, out_features=output_dim, bias=False)

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
        x = torch.reshape(x, (batch_size * sequence_length, self.recurrent_dim))

        A = torch.reshape(
            self.gen_A.forward(x),
            (batch_size, sequence_length, self.state_dim, self.state_dim),
        )

        n = self.state_dim
        m = self.control_dim
        l = math.ceil(n/m)
        K_c = torch.zeros([batch_size, n, l*m], device=control.device)
        B = torch.zeros(batch_size, sequence_length, self.state_dim, self.control_dim)

        "generieren von P"
        gen_P = nn.Linear(
            in_features=self.recurrent_dim, out_features=self.state_dim * self.state_dim, bias=True
        )
        P = torch.reshape(
            gen_P.forward(x),
            (batch_size, self.state_dim, self.state_dim),
        )

        "x[:][:l]"

        "generieren von Q"

        Q = torch.zeros([batch_size, l, n, m], device=control.device)
        gen_Q = nn.Linear(
            in_features=self.recurrent_dim, out_features=self.state_dim * self.control_dim, bias=True
        )

        for batch in range(batch_size):
            for t in range(l):
                Q[batch, t, :, :] = torch.reshape(
                    gen_Q.forward(x[(batch*batch_size)+t, :]),
                    (self.state_dim, self.control_dim),
                )

        "Kopplungsterm"
        kopplung = np.zeros_like(control, shape=[n, l*m])
        kopplung[:, :n] = torch.eye(n)

        temp_K_c = np.dot(P, kopplung)

        "K_c = P * kopplung * Q"
        "temp_K_c = torch.matmul(P, kopplung)"
        "K_c = torch.matmul(temp_K_c, Q)"

        for batch in batch_size:
            for t in range(l):
                "B[batch, t, :, :] = K_c[:][t*m:(t+1)*m-1]"
                B[batch, t, :, :] = np.dot(temp_K_c, B[t])

        B[:, l+1:, :, :] = torch.reshape(
            self.gen_B.forward(x),
            (batch_size, sequence_length-l, self.state_dim, self.control_dim),
        )

        """
        method 3: construction of K_c via multiplication

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
    pass


class ControllableReLiNet2MMModel(SwitchingLSTMBaseModel):
    CONFIG = ControllableReLiNetModelConfig

    def __init__(self, config: ControllableReLiNetModelConfig) -> None:
        if config.switched_system_state_dim is None:
            state_dim = len(config.state_names)
        else:
            state_dim = config.switched_system_state_dim

        predictor = ControllableReLiNet2MM(
            control_dim=len(config.control_names),
            state_dim=state_dim,
            output_dim=len(config.state_names),
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            dropout=config.dropout,
        )
        super().__init__(config=config, predictor=predictor)
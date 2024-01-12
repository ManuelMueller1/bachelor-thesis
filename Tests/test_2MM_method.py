import pathlib

import pipeline

from ControllableModels.TwoMatrixMethod import ControllableReLiNet2MMModel


def test_ControllableReLiNet2MMModel(tmp_path: pathlib.Path) -> None:
    model_name = 'ControllableReLiNet2MM'
    model_class = 'ControllableModels.TwoMatrixMethod.ControllableReLiNet2MMModel'
    config = ControllableReLiNet2MMModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )

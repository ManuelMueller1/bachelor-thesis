import pathlib

import pipeline

from ControllableModels.SVDMethod import ControllableReLiNetSVDModel


def test_ControllableReLiNetSVDModel(tmp_path: pathlib.Path) -> None:
    model_name = 'ControllableReLiNetSVD'
    model_class = 'ControllableModels.SVDMethod.ControllableReLiNetSVDModel'
    config = ControllableReLiNetSVDModel.CONFIG(
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

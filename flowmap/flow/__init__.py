import torch

from ..dataset.types import Batch
from ..misc.nn_module_tools import convert_to_buffer
from .flow_predictor import FlowPredictor, Flows
from .flow_predictor_gmflow import FlowPredictorGMFlow, FlowPredictorGMFlowCfg
from .flow_predictor_raft import FlowPredictorRaft, FlowPredictorRaftCfg

FLOW_PREDICTORS = {
    "gmflow": FlowPredictorGMFlow,
    "raft": FlowPredictorRaft,
}

FlowPredictorCfg = FlowPredictorRaftCfg | FlowPredictorGMFlowCfg


def get_flow_predictor(cfg: FlowPredictorCfg) -> FlowPredictor:
    flow_predictor = FLOW_PREDICTORS[cfg.name](cfg)
    convert_to_buffer(flow_predictor, persistent=False)
    return flow_predictor


@torch.no_grad()
def compute_flows(
    batch: Batch,
    flow_shape: tuple[int, int],
    device: torch.device,
    cfg: FlowPredictorCfg,
) -> Flows:
    print("Precomputing optical flow.")
    flow_predictor = get_flow_predictor(cfg)
    flow_predictor.to(device)
    return flow_predictor.compute_bidirectional_flow(batch.to(device), flow_shape)

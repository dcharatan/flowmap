from .visualizer import Visualizer
from .visualizer_summary import VisualizerSummary, VisualizerSummaryCfg
from .visualizer_trajectory import VisualizerTrajectory, VisualizerTrajectoryCfg

VISUALIZERS = {
    "summary": VisualizerSummary,
    "trajectory": VisualizerTrajectory,
}

VisualizerCfg = VisualizerSummaryCfg | VisualizerTrajectoryCfg


def get_visualizers(cfgs: list[VisualizerCfg]) -> list[Visualizer]:
    return [VISUALIZERS[cfg.name](cfg) for cfg in cfgs]

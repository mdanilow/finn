import os
from os.path import join
import argparse
import shutil

# build steps
from qonnx.core.modelwrapper import ModelWrapper
# streamline
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    ApplyConfig,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
# to hw
from qonnx.transformation.infer_shapes import InferShapes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

# build
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg


def step_yololit_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    model = model.transform(Streamline())
    additional_streamline_transformations = [
        Streamline(),
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastJoinAdd(),
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastJoinAdd(),

        Streamline(),
        LowerConvsToMatMul(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        reorder.MoveTransposePastFork(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        reorder.MoveTransposePastJoinAdd(),
        reorder.MoveTransposePastFork(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        reorder.MoveTransposePastJoinAdd(),
        reorder.MoveTransposePastFork(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        reorder.MoveTransposePastJoinAdd(),
        absorb.AbsorbConsecutiveTransposes()
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
    return model


def step_yololit_convert_to_hw_layers(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferConcatLayer())
    model = model.transform(to_hw.InferSplitLayer())
    model = model.transform(to_hw.InferUpsample())
    model = model.transform(to_hw.InferDuplicateStreamsLayer()) 

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


BUILD_DIR = os.environ["FINN_BUILD_DIR"]
OUTPUT_DIR = join(BUILD_DIR, "yololit_fifosizing")
BOARD = "ZCU104"
# model_file = "yololit320.onnx"
model_file = join(BUILD_DIR, "yolov8_output_dir", "intermediate_models", "step_yololit_convert_to_hw_layers.onnx")
folding_config_file = None
specialize_layers_config_file = None

# which platforms to build the networks for
zynq_platforms = ["ZCU104", "ZCU102"]
alveo_platforms = ["U250"]
# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


build_steps = [
    # step_yololit_streamline,
    # step_yololit_convert_to_hw_layers,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    # "step_create_stitched_ip",
    # # step_slr_floorplan,
    # "step_measure_rtlsim_performance",
    # "step_out_of_context_synthesis",
    # "step_synthesize_bitfile",
    # "step_make_pynq_driver",
    # "step_deployment_package",
]


cfg = build.DataflowBuildConfig(
    output_dir=OUTPUT_DIR,
    verbose=True,
    standalone_thresholds=True,
    folding_config_file=folding_config_file,
    specialize_layers_config_file=specialize_layers_config_file,
    auto_fifo_depths=True,
    # split_large_fifos=True,
    synth_clk_period_ns=10,
    target_fps=90,
    board=BOARD,
    shell_flow_type=platform_to_shell(BOARD),
    steps=build_steps,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)
build.build_dataflow_cfg(model_file, cfg)
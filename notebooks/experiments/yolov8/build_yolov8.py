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
    GiveUniqueNodeNames
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




def step_yolov8_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    model = model.transform(Streamline())
    model = model.transform(reorder.MoveAddPastJoinConcat())
    additional_streamline_transformations = [
        # Affine ops
        reorder.MoveScalarLinearPastSplit(),
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastJoinAdd(),
        reorder.MoveMulPastJoinConcat(),
        Streamline(),
        # Affine ops in SPPF
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastMaxPool(),
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastMaxPool(),
        reorder.MoveMulPastJoinConcat(),
        Streamline(),
        # Transposes
        LowerConvsToMatMul(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        absorb.AbsorbConsecutiveTransposes(),
        reorder.MakeScaleResizeNHWC(),
        reorder.MoveTransposePastSplit(),
        reorder.MoveTransposePastFork(),
        reorder.MoveTransposePastJoinAdd(),
        reorder.MoveTransposePastJoinConcat(),
        absorb.AbsorbConsecutiveTransposes(),
        # Transposes in SPPF
        reorder.MakeMaxPoolNHWC(),
        reorder.MoveTransposePastJoinConcat(),
        absorb.AbsorbConsecutiveTransposes()
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
    return model


def step_yolov8_convert_to_hw_layers(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferPool())
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
OUTPUT_DIR = join(BUILD_DIR, "yolov8_output_dir")
BOARD = "ZCU104"
model_file = "uptoc2f_quantyolov8.onnx"
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

# default_build_dataflow_steps = [
#     "step_qonnx_to_finn",
#     "step_tidy_up",
#     "step_streamline",
#     "step_convert_to_hw",
#     "step_create_dataflow_partition",
#     "step_specialize_layers",
#     "step_target_fps_parallelization",
#     "step_apply_folding_config",
#     "step_minimize_bit_width",
#     "step_generate_estimate_reports",
#     "step_hw_codegen",
#     "step_hw_ipgen",
#     "step_set_fifo_depths",
#     "step_create_stitched_ip",
#     "step_measure_rtlsim_performance",
#     "step_out_of_context_synthesis",
#     "step_synthesize_bitfile",
#     "step_make_pynq_driver",
#     "step_deployment_package",
# ]

build_steps = [
    step_yolov8_streamline,
    step_yolov8_convert_to_hw_layers,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]


#Delete previous run results if exist
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print("Previous run results deleted!")

cfg = build.DataflowBuildConfig(
    output_dir=OUTPUT_DIR,
    verbose=True,
    standalone_thresholds=True,
    folding_config_file=folding_config_file,
    specialize_layers_config_file=specialize_layers_config_file,
    synth_clk_period_ns=10,
    target_fps=30,
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
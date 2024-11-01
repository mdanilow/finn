
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from os.path import join
import time

import cv2

from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device
from yolo import postprocess, plot_one_box


def preprocess(img):
    new_unpad = (320, 180)
    left, right = 0, 0
    top, bottom = 70, 70
    color = (114, 114, 114)

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img = img[:, :, ::-1]

    return np.expand_dims(img, 0)


# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['INT24'], DataType['INT24'], DataType['INT24'], DataType['INT24'], DataType['INT24']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 320, 320, 3)],
    "oshape_normal" : [(1, 20, 20, 255), (1, 10, 10, 255), (1, 5, 5, 255), (1, 3, 3, 255), (1, 2, 2, 255)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 320, 320, 3, 1)],
    "oshape_folded" : [(1, 20, 20, 85, 3), (1, 10, 10, 17, 15), (1, 5, 5, 255, 1), (1, 3, 3, 255, 1), (1, 2, 2, 255, 1)],
    "ishape_packed" : [(1, 320, 320, 3, 1)],
    "oshape_packed" : [(1, 20, 20, 85, 9), (1, 10, 10, 17, 45), (1, 5, 5, 255, 3), (1, 3, 3, 255, 3), (1, 2, 2, 255, 3)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0', 'odma1', 'odma2', 'odma3', 'odma4'],
    "number_of_external_weights": 0,
    "external_weights_input_shapes": {},
    "num_inputs" : 1,
    "num_outputs" : 5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="alveo")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["input.npy"])
    parser.add_argument('--outputfile', help='name(s) of output npy file(s) (i.e. "output.npy")', nargs="*", type=str, default=["output0.npy", "output1.npy", "output2.npy", "output3.npy", "output4.npy"])
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    runtime_weight_dir = args.runtime_weight_dir
    devID = args.device
    device = Device.devices[devID]

    imgdir = 'images'
    img0_shape = (540, 960, 3)  #raw image
    img1_shape = (320, 320, 3)  #after preprocess
    muls = [np.load("Mul_{}.npy".format(i)) for i in range(5)]
    adds = [np.load("Add_{}.npy".format(i)) for i in range(5)]

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir, device=device
    )

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":

        imgnames = os.listdir(imgdir)
        for i, imgname in enumerate(imgnames):
            imgpath = join(imgdir, imgname)

            start_time = time.time()
            img0 = cv2.imread(imgpath)
            img = preprocess(img0)
            preproc_time = time.time() - start_time

            # load desired input .npy file(s)
            ibuf_normal = [img]
            obuf_normal = accel.execute(ibuf_normal)
            accel_time = time.time() - start_time

            preds = postprocess(obuf_normal, muls, adds, img1_shape, img0_shape)
            postproc_time = time.time() - start_time

            # print(preds)
            # draw bboxes
            for *xyxy, conf, cls in reversed(preds):
                plot_one_box(xyxy, img0, color=(0, 0, 255), line_thickness=1)
            cv2.imwrite('outputs/{}'.format(imgname), img0)
            # if not isinstance(obuf_normal, list):
            #     obuf_normal = [obuf_normal]
            # for o, obuf in enumerate(obuf_normal):
            #     np.save(join('test', outputfile[o]), obuf)
            print('pre accel post, total:', preproc_time, accel_time - preproc_time, postproc_time - accel_time, ',', postproc_time)


    elif exec_mode == "throughput_test":
        # remove old metrics file
        try:
            os.remove("nw_metrics.txt")
        except FileNotFoundError:
            pass
        res = accel.throughput_test()
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()
        print("Results written to nw_metrics.txt")
    else:
        raise Exception("Exec mode has to be set to execute or throughput_test")

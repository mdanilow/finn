import json
from typing import List

with open("yololit_fifosizing/final_hw_config.json") as file:
    d = json.load(file)

# MatrixVectorActivation
mva_ramstyle_from_idx = 0
mva_ramstyle = "auto"

mva_restype_from_idx = 0
mva_restype = "dsp"

mva_memmode_from_idx = 0
mva_memmode = "external"

# VectorVectorActivation
vva_ramstyle_from_idx = 0
vva_ramstyle = "auto"

vva_restype_from_idx = 0
vva_restype = "dsp"

vva_memmode_from_idx = 0
vva_memmode = "internal_embedded"

# Thresholding
th_depth_trigger_bram_to_idx = 40
th_depth_trigger_bram = 99999999

# ConvolutionInputGenerator
cig_ramstyle_from_idx = 0
cig_ramstyle = "auto"


for module, module_dict in d.items():
    if("_" in module):
        module_idx = int(module.split("_")[-1])

    if "MVAU" in module:
        if module_idx >= mva_ramstyle_from_idx:
            module_dict["ram_style"] = mva_ramstyle
            if mva_ramstyle == "ultra":
                module_dict["runtime_writeable_weights"] = 1

        if module_idx >= mva_restype_from_idx:
            module_dict["resType"] = mva_restype

        if module_idx >= mva_memmode_from_idx:
            module_dict["mem_mode"] = mva_memmode
        
    elif "VVAU" in module:
        if module_idx >= vva_ramstyle_from_idx:
            module_dict["ram_style"] = vva_ramstyle
            if vva_ramstyle == "ultra":
                module_dict["runtime_writeable_weights"] = 1

        if module_idx >= vva_restype_from_idx:
            module_dict["resType"] = vva_restype

        if module_idx >= vva_memmode_from_idx:
            module_dict["mem_mode"] = vva_memmode

    elif "Thresholding" in module:
        # if module_idx >= th_ramstyle_from_idx:
        #     module_dict["ram_style"] = th_ramstyle
        #     if th_ramstyle == "ultra":
        #         module_dict["runtime_writeable_weights"] = 1
        # if module_idx >= th_memmode_from_idx:
        #     module_dict["mem_mode"] = th_memmode
        if module_idx <= th_depth_trigger_bram_to_idx:
            module_dict["depth_trigger_bram"] = th_depth_trigger_bram

    elif "ConvolutionInputGenerator" in module:
        if module_idx >= cig_ramstyle_from_idx:
            module_dict["ram_style"] = cig_ramstyle


with open('my_folding_config.json', 'w') as file:
        json.dump(d, file, indent=2)



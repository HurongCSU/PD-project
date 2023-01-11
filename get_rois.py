import SimpleITK as sitk
import numpy as np


# 3050\4012\4051\3011
def get_rois():
    seg_temp = sitk.ReadImage('./HarvardOxford-sub-maxprob-thr25-2mm.nii.gz', sitk.sitkFloat32)
    template = sitk.GetArrayFromImage(seg_temp)
    fg_data = np.zeros(template.shape)
    fg_data[np.where(template == 11)] = 1

    template = sitk.GetImageFromArray(fg_data)
    template.CopyInformation(seg_temp)
    sitk.WriteImage(template, './masks/mask11.nii')

get_rois()

import os

import SimpleITK as sitk
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")


def register(fixed, moving):
    # fixed = sitk.Normalize(fixed)
    # fixed = sitk.DiscreteGaussian(fixed, 2.0)
    # moving = sitk.Normalize(moving)
    # moving = sitk.DiscreteGaussian(moving, 2.0)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    R.SetOptimizerScalesFromPhysicalShift()
    final_transform = sitk.AffineTransform(3)
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                              numberOfIterations=200,
                                              convergenceMinimumValue=1e-5,
                                              convergenceWindowSize=5)

    # R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInitialTransform(final_transform)
    R.SetInterpolator(sitk.sitkLinear)

    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)
    """
    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    """

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    """
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # resampler.SetDefaultPixelValue(0)
    # out_m = resampler.Execute(moving_mask)
    out_m = []
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkFloat32)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkFloat32)
    """
    # cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    out = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkFloat32)

    return out


# image.nii, mid_mask.nii
path_t1 = 'F:/PK-PD/PK-PD/'
path_dat = 'F:/spect/spect/PPMI'
ppid = glob(path_dat + '/*/*/*/*/*.dcm')

for dcm in ppid:
    print(dcm)
    name = dcm.split(os.sep)[1]
    print(name)

    template = sitk.ReadImage('./MNI152_T1_2mm.nii.gz', sitk.sitkFloat32)
    datscan = sitk.ReadImage(dcm, sitk.sitkFloat32)

    register_img = register(template, datscan)
    sitk.WriteImage(register_img, dcm.replace('.dcm', '.nii'))

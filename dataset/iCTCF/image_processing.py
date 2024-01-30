import sys
sys.path.append("..")
import os
import re
import seaborn as sns
sns.set()
warnings.filterwarnings('ignore')
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import datetime
import os, glob, json
import cv2
import copy
import seaborn as sns

def resample_image_to_1mm(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_image_to_256x256x256(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [256, 256, 256]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_image_to_128x128x128(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [128, 128, 128]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_128x128x8(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [128, 128, 8]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_256x256x128(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [256, 256, 128]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_xy_target_size(itk_image, out_size, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize() # x, y, z
    out_spacing = [original_spacing[0] / (out_size[2] / original_size[0]), # x
                   original_spacing[1] / (out_size[1] / original_size[1]), # y
                   original_spacing[2] / (original_size[0] / original_size[2])] # z

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize([out_size[1], out_size[2], original_size[2]]) # outsize: z, x, y, setsize: x, y, z
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_xyz_target_size(itk_image, out_size, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize() # x, y, z
    out_spacing = [original_spacing[0] / (out_size[2] / original_size[0]), # x
                   original_spacing[1] / (out_size[1] / original_size[1]), # y
                   original_spacing[2] / (out_size[0] / original_size[2])] # z

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize([out_size[1], out_size[2], out_size[0]]) # outsize: z, x, y, setsize: x, y, z
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def takedate(elem):
    return int(elem)

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def image_compose(npImage_resample_adjust, x=2, z=2):

    [sz,sx,sy] = npImage_resample_adjust.shape
    combine_x = np.zeros([x * sz, z * sy])
    combine_z = np.zeros([x * sx, z * sy])

    total_num = 0
    for ix in range(x):
        for iz in range(z):
            print(ix,iz)
            combine_x[ix*sz:(ix+1)*sz, iz*sy:(iz+1)*sy] = npImage_resample_adjust[::-1,sx//5*(ix*x+iz+1),::]
            combine_z[ix*sx:(ix+1)*sx, iz*sy:(iz+1)*sy] = npImage_resample_adjust[sz//5*(ix*x+iz+1),:,::]

    return combine_x, combine_z

def image_3D_normalisation(npImage, min_value=-1024, max_value=-100):

    # crop
    npImage_norm = npImage
    npImage_norm[npImage < min_value] = min_value
    npImage_norm[npImage > max_value] = max_value

    npImage_norm = (npImage_norm-min_value)/(max_value-min_value)


    return npImage_norm
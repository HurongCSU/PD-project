import os
from glob import glob
from utils.extract_features import get_features
from utils.table import list_csv
import SimpleITK as sitk


def get_modality_feature(patient, modality):
    return get_features(os.path.join(patient, modality, 'image.nii'),
                        os.path.join(patient, modality, 'mid_mask.nii'))


def correct_bias(in_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using SimpleITK N4BiasFieldCorrection.
    :param in_file: .nii.gz 文件的输入路径
    :param out_file: .nii.gz 校正后的文件保存路径
    :return: 校正后的nii文件全路径名

    """
    # 使用SimpltITK N4BiasFieldCorrection校正MRI图像的偏置场
    input_image = sitk.ReadImage(in_file, image_type)
    output_image_s = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    return output_image_s


# image path
path = ['F:/PK-PD/PK-PD']
n = 129
for pp in path:
    patients = glob(pp + '/*')

    t1_dataset = []
    t2_dataset = []

    feature_name, _ = get_modality_feature(patients[0], 'T1')
    feature_name.insert(0, 'name')

    for pt in patients:
        print(pt)
        temp = [pt]
        try:
            _, t1c = get_modality_feature(pt, 'T1')
            temp.extend(t1c)
        except:
            temp.extend(['NAN'] * n)

        t1_dataset.append(temp)

        temp = [pt]
        try:
            _, t2f = get_modality_feature(pt, 'T2')
            temp.extend(t2f)
        except:
            temp.extend(['NAN'] * n)
        t2_dataset.append(temp)

    list_csv(t1_dataset, os.path.join(pp, 't1c_original.csv'), column=feature_name)
    list_csv(t2_dataset, os.path.join(pp, 't2f_original.csv'), column=feature_name)

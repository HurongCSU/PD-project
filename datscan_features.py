import os
from glob import glob
from utils.extract_features import get_features
from utils.table import list_csv


def get_modality_feature(patient, num):
    if num == 5:
        return get_features(patient,
                        '../masks/mask5.nii')
    if num == 6:
        return get_features(patient,
                        '../masks/mask6.nii')
    if num == 16:
        return get_features(patient,
                        '../masks/mask16.nii')
    if num == 17:
        return get_features(patient,
                        '../masks/mask17.nii')

    if num == 21:
        return get_features(patient,
                        '../masks/mask21.nii')

    if num == 11:
        return get_features(patient,
                        '../masks/mask11.nii')


# image path
path = 'F:/spect/spect/PPMI'
n = 129
patients = glob(path + '/*/*/*/*/*.nii')

pet5_dataset = []
pet6_dataset = []
pet16_dataset = []
pet17_dataset = []
pet21_dataset = []
pet11_dataset = []

feature_name, _ = get_modality_feature(patients[0], 5)
feature_name.insert(0, 'name')

for pt in patients:
    name = pt.split(os.sep)[1]
    print(name)
    temp = [name]
    try:
        _, pet = get_modality_feature(pt, 5)
        temp.extend(pet)
    except:
        temp.extend(['0'] * n)

    pet5_dataset.append(temp)

    temp = [name]
    try:
        _, pet = get_modality_feature(pt, 6)
        temp.extend(pet)
    except:
        temp.extend(['0'] * n)
    pet6_dataset.append(temp)

    temp = [name]
    try:
        _, pet = get_modality_feature(pt, 16)
        temp.extend(pet)
    except:
        temp.extend(['0'] * n)
    pet16_dataset.append(temp)

    temp = [name]
    try:
        _, pet = get_modality_feature(pt, 17)
        temp.extend(pet)
    except:
        temp.extend(['0'] * n)
    pet17_dataset.append(temp)

    temp = [name]
    try:
        _, pet = get_modality_feature(pt, 21)
        temp.extend(pet)
    except:
        temp.extend(['0'] * n)
    pet21_dataset.append(temp)

    temp = [name]
    try:
        _, pet = get_modality_feature(pt, 11)
        temp.extend(pet)
    except:
        temp.extend(['0'] * n)
    pet11_dataset.append(temp)

list_csv(pet5_dataset, os.path.join(path, 'pet5.csv'), column=feature_name)
list_csv(pet6_dataset, os.path.join(path, 'pet6.csv'), column=feature_name)
list_csv(pet16_dataset, os.path.join(path, 'pet16.csv'), column=feature_name)
list_csv(pet17_dataset, os.path.join(path, 'pet17.csv'), column=feature_name)
list_csv(pet21_dataset, os.path.join(path, 'pet21.csv'), column=feature_name)
list_csv(pet11_dataset, os.path.join(path, 'pet11.csv'), column=feature_name)

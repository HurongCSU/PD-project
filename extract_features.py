"""
pyradiomics 文档
https://pyradiomics.readthedocs.io/en/latest/customization.html#radiomics-image-types-label
"""

from __future__ import print_function
from radiomics import featureextractor



# 返回提取的特征名和值->列表
def get_features(img_path, mask_path):
    print(img_path, mask_path)
    extractor = featureextractor.RadiomicsFeatureExtractor('./Params.yaml')
    # 提取所有特征
    extractor.enableAllFeatures()
    print(img_path, "Calculating features")
    featureVector = extractor.execute(img_path, mask_path)

    for featureName in featureVector.keys():
        print("Computed %s: %s" % (featureName, featureVector[featureName]))

    return list(featureVector.keys()), list(featureVector.values())



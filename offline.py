# import glob
# import os
# import pickle
# from PIL import Image
# from feature_extractor import FeatureExtractor

# fe = FeatureExtractor()

# for img_path in sorted(glob.glob('static/img/*.jpg')):
#     print(img_path)
#     img = Image.open(img_path)  # PIL(Python Imaging Library) image
#     feature = fe.extract(img)
#     feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
#     pickle.dump(feature, open(feature_path, 'wb'))

import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
import numpy as np
fe = FeatureExtractor()
features=[]
images=[]
for img_path in sorted(glob.glob('static/img/*.jpg')):
    print(img_path)
    img = Image.open(img_path)  # PIL(Python Imaging Library) image
    feature = fe.extract(img)
    feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
    features.append(feature)
    print(img_path)
    images.append(img_path)
np.save("features.npy",np.array(features))
np.save("images.npy",np.array(images))
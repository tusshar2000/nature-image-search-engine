from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.models import Model
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D


class FeatureExtractor:
    def __init__(self):
        #sbase_model =
        res_conv =  MobileNet(weights='imagenet',include_top=False)
        #print(self.model)
        x = res_conv.output
        x = GlobalAveragePooling2D()(x)
	#fully-connected layer
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(6, activation='softmax')(x)
        # this is the model we will train
        self.model = Model(inputs=res_conv.input, outputs=predictions)
        self.model.load_weights("weights.h5")
        for layer in self.model.layers:
            print(layer.name)
        self.model = Model(inputs=res_conv.input, outputs=self.model.get_layer('dense_1').output)
        self.model._make_predict_function()
        
    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        feature = self.model.predict(x).flatten() # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize




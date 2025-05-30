import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm
import pickle

# Load InceptionV3 model (remove the last layer)
model = InceptionV3(weights='imagenet')
model = Model(inputs=model.input, outputs=model.layers[-2].output)

def extract_features(img_folder):
    features = {}
    for img_name in tqdm(os.listdir(img_folder)):
        file_path = os.path.join(img_folder, img_name)
        try:
            img = image.load_img(file_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            features[img_name] = feature.reshape(2048,)
        except:
            continue
    return features

if __name__ == "__main__":
    img_folder = "../dataset/Flicker8k_Dataset"
    features = extract_features(img_folder)
    print(f"Extracted features for {len(features)} images.")
    
    with open("../dataset/image_features.pkl", "wb") as f:
        pickle.dump(features, f)

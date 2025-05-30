import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# 🔹 Load tokenizer from file
def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 🔹 Extract image features using pre-trained InceptionV3
def extract_image_features(img_path):
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    return features[0]

# 🔹 Generate a caption from image features
def generate_caption(model, tokenizer, photo, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([np.array([photo]), sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break

        if word is None:
            break

        in_text += ' ' + word
        if word == '<end>':
            break

    final_caption = in_text.replace('<start>', '').replace('<end>', '').strip()
    return final_caption

# 🔹 Main script to test image captioning
if __name__ == "__main__":
    print("🔹 Loading model and tokenizer...")
    model = load_model("../dataset/caption_model.keras")
    tokenizer = load_tokenizer("../dataset/tokenizer.pkl")

    # ✅ Ensure this matches training value
    max_length = 37

    test_img_path = "../test1.jpg"
    if not os.path.exists(test_img_path):
        print(f"❌ File not found: {test_img_path}")
    else:
        print("🔹 Extracting image features...")
        photo_features = extract_image_features(test_img_path)

        print("🔹 Generating caption...")
        caption = generate_caption(model, tokenizer, photo_features, max_length)

        print("\n🖼️ Caption:")
        print(f"👉 {caption}")

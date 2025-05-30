import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# 🔹 Load tokenizer from file
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as file:
        return pickle.load(file)

# 🔹 Extract features from the uploaded image
def extract_features(img_path):
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    return features[0]

# 🔹 Generate caption based on image features
def generate_caption(model, tokenizer, photo_features, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        prediction = model.predict([np.array([photo_features]), sequence], verbose=0)
        predicted_id = np.argmax(prediction)

        word = next((w for w, idx in tokenizer.word_index.items() if idx == predicted_id), None)
        if word is None:
            break

        in_text += ' ' + word
        if word == '<end>':
            break

    final_caption = in_text.replace('<start>', '').replace('<end>', '').strip()
    return final_caption

# 🔷 Streamlit App
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")
st.title("🖼️ AI-Powered Image Caption Generator")

uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="📸 Uploaded Image", use_column_width=True)

    try:
        st.info("🔄 Loading model and tokenizer...")
        model = load_model("dataset/caption_model.keras")
        tokenizer = load_tokenizer("dataset/tokenizer.pkl")
        max_length = 37  # Make sure this matches training configuration

        st.info("🔍 Extracting image features...")
        photo_features = extract_features(img_path)

        st.info("🧠 Generating caption...")
        caption = generate_caption(model, tokenizer, photo_features, max_length)

        st.subheader("📝 Generated Caption:")
        st.success(caption)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

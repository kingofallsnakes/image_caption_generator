import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load cleaned captions
def load_cleaned_captions(filepath):
    captions = {}
    with open(filepath, 'r') as f:
        for line in f:
            img_id, caption = line.strip().split('\t')
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    return captions

# Get all captions into one list
def get_all_captions(captions_dict):
    all_captions = []
    for cap_list in captions_dict.values():
        all_captions.extend(cap_list)
    return all_captions

# Create tokenizer
def create_tokenizer(captions_dict):
    captions_list = get_all_captions(captions_dict)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions_list)
    return tokenizer

# Save tokenizer
def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

# Create training sequences (no one-hot encoding)
def create_sequences(tokenizer, max_length, captions_dict, image_features):
    X1, X2, y = [], [], []
    for img_id, caption_list in captions_dict.items():
        feature = image_features.get(img_id)
        if feature is None:
            continue
        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)  # Not one-hot, just label
    return shuffle(np.array(X1), np.array(X2), np.array(y, dtype='int32'))

# Define the model
def define_model(vocab_size, max_length):
    # Feature extractor (image)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence processor (text)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (merge)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

# Main
if __name__ == "__main__":
    print("🔹 Loading captions and image features...")
    captions_path = "../dataset/cleaned_captions.txt"
    features_path = "../dataset/image_features.pkl"

    captions_dict = load_cleaned_captions(captions_path)
    with open(features_path, "rb") as f:
        image_features = pickle.load(f)

    print("🔹 Creating tokenizer...")
    tokenizer = create_tokenizer(captions_dict)
    save_tokenizer(tokenizer, "../dataset/tokenizer.pkl")

    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in get_all_captions(captions_dict))
    print(f"✅ Vocabulary Size: {vocab_size}")
    print(f"✅ Max Caption Length: {max_length}")

    print("🔹 Preparing training data (may take some time)...")
    X1, X2, y = create_sequences(tokenizer, max_length, captions_dict, image_features)

    # Optional: Save training data
    np.save("../dataset/X1.npy", X1)
    np.save("../dataset/X2.npy", X2)
    np.save("../dataset/y.npy", y)

    print("✅ Training data prepared!")

    print("🔹 Building model...")
    model = define_model(vocab_size, max_length)
    model.summary()

    print("🔹 Training model...")
    model.fit([X1, X2], y, epochs=20, batch_size=64, verbose=1)

    model.save("../dataset/caption_model.keras")
    print("✅ Model trained and saved as caption_model.keras ")

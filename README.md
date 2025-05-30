
# 🖼️ Image Caption Generator using Python (CNN + LSTM)

This project uses deep learning techniques (Convolutional Neural Networks + LSTM) to automatically generate captions for images — just like a human!

---

## 📌 Project Description

Humans can look at an image and instantly understand its context. Can a computer do the same? With recent advancements in deep learning, this is possible.

This project demonstrates how to train a model to generate meaningful captions for images using:
- **CNN (InceptionV3)**: For image feature extraction
- **LSTM (RNN)**: For sequence modeling to generate captions


## 📂 Folder Structure

```

image_caption_generator
│   .gitattributes
│   app.py
│
├───dataset
│   │   caption_model.h5
│   │   caption_model.keras
│   │   cleaned_captions.txt
│   │   Flickr8k.token.txt
│   │   image_features.pkl
│   │   tokenizer.pkl
│   │
│   └───Flickr8k_text
│       │   CrowdFlowerAnnotations.txt
│       │   ExpertAnnotations.txt
│       │   Flickr8k.lemma.token.txt
│       │   Flickr8k.token.txt
│       │   Flickr_8k.devImages.txt
│       │   Flickr_8k.testImages.txt
│       │   Flickr_8k.trainImages.txt
│       │   
│       │
│       └───__MACOSX
│               ._Flickr8k.lemma.token.txt
│
└───src
        caption_generator.py
        caption_preprocessing.py
        feature_extraction.py
        model_training.py

```

## 🛠️ Installation

### ✅ Step 1: Clone the repo

```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
````

### ✅ Step 2: Install dependencies

```bash
pip install tensorflow keras numpy pillow matplotlib tqdm pickle5
```

### ✅ Step 3: Download Dataset

* [Flickr8k Images](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
* [Flickr8k Captions](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

Place them in the `dataset/` folder.

---

## 🚀 How to Run

### 🔹 Step 1: Extract features from images

```python
from utils.image_preprocessing import extract_features
features = extract_features("dataset/Flickr8k_Dataset")
# Save to disk using pickle
```

### 🔹 Step 2: Preprocess captions and fit tokenizer

```python
from utils.data_preprocessing import clean_caption
# Process captions, build tokenizer, and save it
```

### 🔹 Step 3: Train the model

```bash
python app.py
```

---

## 🤖 How It Works

1. Extract features from images using InceptionV3.
2. Clean and process text captions.
3. Train a combined CNN + LSTM model on image-text pairs.
4. Generate captions for new images.

---

## 🧪 Example Output

> 🖼️ Input: A dog running in the park
> 📝 Output: `"startseq a dog is running through the grass endseq"`

---

## 📦 Requirements

* Python 3.7+
* TensorFlow 2.x
* NumPy, Pillow, Matplotlib, tqdm, nltk
* OpenCV (optional for UI)

---

## 📈 Future Improvements

* Beam Search for better caption generation
* Streamlit UI for image upload & display
* Support for larger datasets (e.g., MS-COCO)

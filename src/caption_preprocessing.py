import string
import re
from tqdm import tqdm

# Load captions from file
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, caption = line.split('\t')
            img_id = img_id.split('#')[0]
            caption = caption.lower()
            caption = re.sub(r'[^a-z ]', '', caption)  # remove punctuation/numbers
            caption = 'startseq ' + caption + ' endseq'
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    return captions

# Save the cleaned captions to file
def save_cleaned_captions(captions, output_path):
    lines = []
    for img_id, caps in captions.items():
        for cap in caps:
            lines.append(f"{img_id}\t{cap}")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    input_path = "../dataset/Flickr8k.token.txt"
    output_path = "../dataset/cleaned_captions.txt"

    print("Loading and cleaning captions...")
    cleaned = load_captions(input_path)

    print(f"Total images with captions: {len(cleaned)}")
    save_cleaned_captions(cleaned, output_path)
    print("Saved cleaned captions.")

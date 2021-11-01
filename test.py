import pathlib
import pandas as pd
import torch

from model.model import encoder, decoder
from data_loader.data_loader import collate_fn
from data_loader.data_loader import dataLoader

from torchvision import transforms
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image
from nltk.tokenize import word_tokenize


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

root = pathlib.Path('./datasets')

pathData = root / 'captions.txt'

with open(pathData) as f:
    npData = pd.read_csv(f).to_numpy()

fileNames = npData[:, 0:5][::5] # 1 img have 5 captions

captions = npData[:, 1]

sizeCaps = len(captions)

counter = Counter()

for caption in captions:
    tokens = word_tokenize(caption.lower())
    counter.update(tokens)

words = []
words.append('<pad>')
words.append('<start>')
words.append('<end>')
words.append('<unk>')
words.extend([word for word, cnt in counter.items() if cnt >= 1])


word2idx = {}
idx2word = {}
idx = 0

for i, word in enumerate(words):
    word2idx[word] = i
    idx2word[i] = word

caps = []
for cap in captions:
    tmp = []
    tokens = word_tokenize(cap.lower())
    tmp.extend([word2idx[tok] for tok in tokens])
    caps.append(tmp)

dataset = dataLoader(root=root, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True,
                                          collate_fn=collate_fn)

vocab_size = len(dataset.words)
embed_size = 512
hidden_size = 512
batch_size = 1

encoder = encoder().eval().to(device)
decoder = decoder(vocab_size=vocab_size,
                  embed_size=embed_size,
                  hidden_size=hidden_size,
                  num_layers=1,
                  max_seq_length=dataset.maxLen).to(device)


encoder_path = './en_1030.pth'
decoder_path = './de_1030.pth'

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

#################################

imagePath = pathlib.Path('./image_4.jpg')

image = load_image(imagePath, transform)
image_tensor = image.to(device)

feature = encoder(image_tensor)
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

sampled_caption = []
for word_id in sampled_ids:
    word = idx2word[word_id]
    if word == '<start>':
        continue
    if word == '<end>':
        break
    sampled_caption.append(word)

sentence = ' '.join(sampled_caption)
print(sentence)
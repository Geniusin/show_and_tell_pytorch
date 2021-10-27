import torch
import pathlib
import numpy as np
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] # resnet 마지막에 붙어있는 fc 제거

        self.resnet = nn.Sequential(*modules) # 리스트로 변환된 resnet의 모듈들을 하나로 묶음
        self.linear = nn.Linear(resnet.fc.in_features, 512)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1) # Flatten
        features = self.linear(features)
        return features


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self,  captions, feature):
        """Decode image feature vectors and generates captions."""

        embeddings = self.embed(captions)
        inputs = torch.cat((feature.unsqueeze(0), embeddings), dim=0)

        hiddens, (h, c) = self.lstm(inputs)

        out_d = self.linear(hiddens)

        return hiddens, (h, c)

sentence = 'l love you so much'

sen = sentence.split()

tok_sen = torch.LongTensor([2, 4, 3, 11, 32])



voca_size = 100

model = RNN(embed_size=10, vocab_size=voca_size, hidden_size=512, num_layers=1)
print(f"input_dim = {tok_sen.size()}")


img = torch.randn(1, 3, 224, 224)

CNN = CNN()

f = CNN(img)
feature = f.unsqueeze(1)

embed, (h, c) = model(tok_sen, feature)
print(f"output_dim = {embed.size()}")


vocab_size = 100
embed_size = 512
output_size = 256
length = 5
batch_size = 1

feature = torch.randn(1, 1, embed_size)
caption = torch.LongTensor([[2], [4], [3], [11], [32]])

h_0 = torch.zeros(1, batch_size, output_size)
c_0 = torch.zeros(1, batch_size, output_size)

embedding = nn.Embedding(vocab_size, embed_size)
TestLSTM = nn.LSTM(embed_size, output_size, 1)
dense = nn.Linear(output_size, vocab_size)
softmax = nn.Softmax(dim=2)

for i in range(10):

    for cap in caption:

        word = embedding(cap)
        out_feature, (h_n, c_n) = TestLSTM(feature, (h_0, c_0))
        out, (h_n, c_n) = TestLSTM(word.unsqueeze(0), (h_n, c_n))
        out_d = dense(out)

        out_s = softmax(out_d)
        print(f'out_s.size = {out_s.size()}')

        criterion = nn.NLLLoss()


        loss = criterion(out_s.squeeze(axis=1), cap)
        loss.backward()

    optimizer = torch.optim.Adam(TestLSTM.parameters(), lr=0.0001)
    optimizer1 = torch.optim.Adam(dense.parameters(), lr=0.0001)
    optimizer2 = torch.optim.Adam(embedding.parameters(), lr=0.0001)

    optimizer.step()
    optimizer.zero_grad()

    optimizer1.step()
    optimizer1.zero_grad()

    optimizer2.step()
    optimizer2.zero_grad()

    print(list(TestLSTM.parameters())[0][0][0])
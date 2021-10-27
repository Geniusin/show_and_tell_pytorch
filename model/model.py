import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence



class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] # resnet 마지막에 붙어있는 fc 제거

        self.resnet = nn.Sequential(*modules) # 리스트로 변환된 resnet의 모듈들을 하나로 묶음
        self.linear = nn.Linear(resnet.fc.in_features, 512)
        self.bn = nn.BatchNorm1d(512, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1) # Flatten
        features = self.linear(features)  # [, 512)
        features = self.bn(features)

        return features


class decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, max_seq_length=38):
        super(decoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,  captions, feature, lengths):
        """Decode image feature vectors and generates captions."""

        embeddings = self.embedding(captions) # [n, 512] [len, embed_size]
        inputs = torch.cat((feature.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(inputs, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        out_d = self.linear(hiddens[0])

        output = self.softmax(out_d)

        return output

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

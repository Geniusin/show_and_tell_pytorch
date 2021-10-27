import pathlib
import numpy as np

from data_loader.data_loader import dataLoader
from data_loader.data_loader import collate_fn
from model.model import encoder, decoder

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

root = pathlib.Path('./datasets')


dataset = dataLoader(root=root, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True,
                                          collate_fn=collate_fn)
vocab_size = len(dataset.words)


embed_size = 512
hidden_size = 512
batch_size = 64

encoder = encoder().to(device)
decoder = decoder(vocab_size=vocab_size,
                  embed_size=embed_size,
                  hidden_size=hidden_size,
                  num_layers=1).to(device)

criterion = nn.NLLLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

total_step = len(data_loader)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

EPOCHS = 1
for epoch in range(EPOCHS):
    start.record()
    for i, (images, captions, lengths) in enumerate(data_loader):

        images = images.to(device)
        captions = captions.squeeze(1).to(device)

        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        feature = encoder(images)

        out = decoder(captions, feature, lengths)

        loss = criterion(out, targets)

        decoder.zero_grad()
        encoder.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, EPOCHS, i, total_step, loss.item(), np.exp(loss.item())))

    if epoch % 10 == 0:
        torch.save(encoder.state_dict(), f'./en_ver5_epoch{epoch}_fin.pth')
        torch.save(decoder.state_dict(), f'./de_ver5_epoch{epoch}_fin.pth')

    end.record()
    torch.cuda.synchronize()

    print(f'{start.elapsed_time(end) / 1000} sec per epoch')

torch.save(encoder.state_dict(), f'./en.pth')
torch.save(decoder.state_dict(), f'./de.pth')

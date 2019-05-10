import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# Counting task

# Need to generate binary numbers
# [0, 0, 1], [0, 1, 0], [0, 1, 1]

# (sequence length, batch_size, dims)

class SequenceTask(object):
    def __init__(self, bits=16):
        self.bits = bits
        self.max_num = 2**bits
        self.all_numbers = torch.arange(self.max_num)

    def to_binary(self, x):
        bins = map(int, list(bin(x))[2:])
        assert len(bins) <= self.bits
        bins = [0]*(self.bits-len(bins)) + bins
        return bins

    def get_batch(self, batch_size, sequence_length, train):
        # train is ignored now
        starts = torch.randint(self.max_num - sequence_length, size=(batch_size,))

        batch = []
        for t in xrange(sequence_length):
            slice = []
            for i in xrange(batch_size):
                x = starts[i] + t
                binary = self.to_binary(x)
                slice.append(binary)
            batch.append(slice)

        batch = torch.LongTensor(batch)

        input = batch[:-1, :, : ].float()
        target = batch[1:, :, :].float()

        return input, target

    def number_to_batch(self, x):
        bins = self.to_binary(x)
        return torch.Tensor([[bins]])

    def to_number(self, bins):
        bins = ''.join(map(str,bins.long().tolist()))
        return int(bins, 2)

    def sample_output(self, output):
        sampled_output = (torch.rand(output.size()) < output).float()
        return sampled_output

    def greedy_output(self, output):
        greedy_output = (output + 0.5).long().float()
        return greedy_output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nbits, nhid, nlayers=2, dropout=0.5):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(nbits, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, nbits)
        self.nbits = nbits
        self.nhid = nhid

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        output = output.view(-1, self.nhid)
        output = self.decoder(output)
        output = output.view(input.size()[0], input.size()[1], self.nbits)
        output = torch.sigmoid(output)
        return output, hidden

    def init_hidden(self, bsz):
        # Wuuuut ?
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


iterations = 10000
bits = 16
batch_size = 10
sequence_length = 10
lr = 20
clip = 0.25
test = True

task = SequenceTask(bits)
model = RNNModel(nbits=bits, nhid=64, nlayers=2)
criterion = nn.BCELoss()

if not test:

    total_loss = []
    accuracies = []
    for i in xrange(iterations):

        # Get new batch from training
        input_, target = task.get_batch(batch_size, sequence_length, train=True)


        model.zero_grad()

        hidden = None

        output, hidden = model(input_, hidden)

        loss = criterion(output, target)
        loss.backward()

        # Get accuracy
        sampled_output = (torch.rand(output.size()) < output).float()
        accuracy = (sampled_output == target).float().mean()
        accuracies.append(accuracy.item())

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss.append(loss.item())

        if i % 100 == 0:
            print 'Iteration', i
            print 'Average loss', np.mean(total_loss)
            print 'Average accuracy', np.mean(accuracies)

        if i % 1000 == 0:
            print 'Save model'
            torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))


# Compare no memory with memory
model.eval()
hidden = None
acc_mem = []
acc_nomem = []
for i in xrange(10000):
    batch = task.number_to_batch(i)
    target = task.number_to_batch(i+1)

    output, hidden = model(batch, hidden) # memory
    greedy_output = task.greedy_output(output)
    acc = (greedy_output == target).float().mean()
    acc_mem.append(acc)

    output, __ = model(batch) # no memory
    greedy_output = task.greedy_output(output)
    acc = (greedy_output == target).float().mean()
    acc_nomem.append(acc)

    if i % 100 == 0:
        print 'i', i
        print 'Mem', np.mean(acc_mem)
        print 'Nomem', np.mean(acc_nomem)


hidden = None
keep = True
while True:
    model.eval()
    number = input('Give number > ')
    batch = task.number_to_batch(number)
    if not keep:
        hidden = None
    output, hidden = model(batch, hidden)
    sampled_output = task.sample_output(output)
    greedy_output = task.greedy_output(output)
    print 'Prob output', output
    print '\nSampled output', sampled_output
    print 'Sampled Number', task.to_number(sampled_output.view(-1))
    print 'Greedy Number', task.to_number(greedy_output.view(-1))


    pass
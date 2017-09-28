# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch
import copy
import os
import random

from .modules import layers2


class GlobalAttnS2SAgent(Agent):
    """Simple agent which uses an RNN to process incoming text observations.
    The RNN generates a vector which is used to represent the input text,
    conditioning on the context to generate an output token-by-token.

    For more information, see Sequence to Sequence Learning with Neural Networks
    `(Sutskever et al. 2014) <https://arxiv.org/abs/1409.3215>`_.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=64,
            help='size of the hidden layers and embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
            help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5,
            help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
            help='dropout rate')
        agent.add_argument('--no-cuda', action='store_true', default=True,
            help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)
        if not shared:
            # this is not a shared instance of this class, so do full
            # initialization. if shared is set, only set up shared members.
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, self.states = self.load(opt['model_file'])
                # override options with stored ones
                opt = self.override_opt(new_opt)

            self.dict = DictionaryAgent(opt)
            self.id = 'Seq2Seq'
            # we use END markers to break input and output and end our output
            self.START = self.dict.start_token
            self.END = self.dict.end_token
            self.observation = {'text': self.END, 'episode_done': True}
            self.END_TENSOR = torch.LongTensor(self.dict.parse(self.END))
            self.START_TENSOR = torch.LongTensor(self.dict.parse(self.START))

            # store important params directly
            hsz = opt['hiddensize']
            self.hidden_size = hsz
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learningrate']
            self.longest_label = 1

            # set up modules
            self.criterion = nn.NLLLoss()
            # lookup table storesemb word embeddings
            self.lt = nn.Embedding(len(self.dict), hsz, padding_idx=0,
                                   scale_grad_by_freq=True)
            # encoder captures the input text
            self.encoder = layers2.EncoderRNN(len(self.dict), hsz, opt['numlayers'])
            # decoder produces our output states
            self.decoder = layers2.AttnDecoderRNN(hsz, len(self.dict), opt['numlayers'])
            # linear layer helps us produce outputs from final decoder state
            self.h2o = nn.Linear(hsz, len(self.dict))
            # droput on the linear layer helps us generalize
            self.dropout = nn.Dropout(opt['dropout'])
            # softmax maps output scores to probabilities
            self.softmax = nn.LogSoftmax()

            # set up optims for each module
            lr = opt['learningrate']
            self.optims = {
                'lt': optim.SGD(self.lt.parameters(), lr=lr),
                'encoder': optim.SGD(self.encoder.parameters(), lr=lr),
                'decoder': optim.SGD(self.decoder.parameters(), lr=lr),
                'h2o': optim.SGD(self.h2o.parameters(), lr=lr),
            }

            if hasattr(self, 'states'):
                # set loaded states if applicable
                self.set_states(self.states)

            # check for cuda
            self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
            self.use_cuda = False
            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])
            if self.use_cuda:
                self.cuda()

        self.episode_done = True

    def override_opt(self, new_opt):
        """Print out each added key and each overriden key."""
        for k, v in new_opt.items():
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def parse(self, text):
        return torch.LongTensor(self.dict.txt2vec(text))

    def v2t(self, vec):
        return self.dict.vec2txt(vec)

    def cuda(self):
        self.START_TENSOR = self.START_TENSOR.cuda(async=True)
        self.END_TENSOR = self.END_TENSOR.cuda(async=True)
        self.criterion.cuda()
        self.lt.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.h2o.cuda()
        self.dropout.cuda()
        self.softmax.cuda()

    def hidden_to_idx(self, hidden, dropout=False):
        """Converts hidden state vectors into indices into the dictionary."""
        if hidden.size(0) > 1:
            raise RuntimeError('bad dimensions of tensor:', hidden)
        hidden = hidden.squeeze(0)
        scores = self.h2o(hidden)
        if dropout:
            scores = self.dropout(scores)
        scores = self.softmax(scores)
        _max_score, idx = scores.max(1)
        return idx, scores

    def zero_grad(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        for optimizer in self.optims.values():
            optimizer.step()

    def init_zeros(self, bsz=1):
        t = torch.zeros(self.num_layers, bsz, self.hidden_size)
        if self.use_cuda:
            t = t.cuda(async=True)
        return Variable(t)

    def init_rand(self, bsz=1):
        t = torch.FloatTensor(self.num_layers, bsz, self.hidden_size)
        t.uniform_(0.05)
        if self.use_cuda:
            t = t.cuda(async=True)
        return Variable(t)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def predict(self, xs, ys=None):
        encoder_hidden = self.encoder.initHidden()
        max_length = 2048


        batchsize = len(xs)
        input_length = xs.size()[1]
        # print(xs)
        # print(input_length)
        target_length = ys.size()[1]
        # print(ys)
        # print(target_length)
        # exit()
        encoder_output = 0
        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder.forward(
                xs.t()[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(self.START_TENSOR)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True
        output_lines = [[] for _ in range(batchsize)]

        if ys is not None:
            # update the model based on the labels
            self.zero_grad()
            loss = 0
            # keep track of longest label we've ever seen
            self.longest_label = max(self.longest_label, ys.size(1))
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder.forward(
                        decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                    loss += self.criterion(decoder_output, ys.t()[di])
                    decoder_input = ys.t()[di]  # Teacher forcing

            # else:
            #     # Without teacher forcing: use its own predictions as the next input
            #     for di in range(target_length):
            #         decoder_output, decoder_hidden, decoder_attention = decoder(
            #             decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            #         topv, topi = decoder_output.data.topk(1)
            #         ni = topi[0][0]
            #
            #         decoder_input = Variable(torch.LongTensor([[ni]]))
            #         decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            #
            #         loss += criterion(decoder_output[0], target_variable[di])
            #         if ni == EOS_token:
            #             break
                    _, preds = decoder_output.max(1)
                    # convert the output scores to tokens
                    token = self.v2t([preds.data[0]])
                    output_lines[0].append(token)

            if random.random() < 0.1:
                true = self.v2t(ys.data[0])
                print('loss:', round(loss.data[0]/target_length, 2),
                      ' '.join(output_lines[0]), '(true: {})'.format(true))

            loss.backward()
            self.update_params()
        # else:
        #     # just produce a prediction without training the model
        #     done = [False for _ in range(batchsize)]
        #     total_done = 0
        #     max_len = 0
        #
        #     while(total_done < batchsize) and max_len < self.longest_label:
        #         # keep producing tokens until we hit END or max length for each
        #         # example in the batch
        #         output, hn = self.decoder(xes, hn)
        #         preds, scores = self.hidden_to_idx(output, dropout=False)
        #         xes = self.lt(preds.t())
        #         max_len += 1
        #         for b in range(batchsize):
        #             if not done[b]:
        #                 # only add more tokens for examples that aren't done yet
        #                 token = self.v2t(preds.data[b])
        #                 if token == self.END:
        #                     # if we produced END, we're done
        #                     done[b] = True
        #                     total_done += 1
        #                 else:
        #                     output_lines[b].append(token)
        #     if random.random() < 0.1:
        #         print('prediction:', ' '.join(output_lines[0]))

        return output_lines


    def batchify(self, observations):
        """Convert a list of observations into input & target tensors."""
        # valid examples
        exs = [ex for ex in observations if 'text' in ex]
        # the indices of the valid (non-empty) tensors
        valid_inds = [i for i, ex in enumerate(observations) if 'text' in ex]

        # set up the input tensors
        batchsize = len(exs)
        # tokenize the text
        parsed = [self.parse(ex['text']) for ex in exs]
        max_x_len = max([len(x) for x in parsed])
        xs = torch.LongTensor(batchsize, max_x_len).fill_(0)
        # pack the data to the right side of the tensor for this model
        for i, x in enumerate(parsed):
            offset = max_x_len - len(x)
            for j, idx in enumerate(x):
                xs[i][j + offset] = idx
        if self.use_cuda:
            xs = xs.cuda(async=True)
        xs = Variable(xs)

        # set up the target tensors
        ys = None
        if 'labels' in exs[0]:
            # randomly select one of the labels to update on, if multiple
            # append END to each label
            labels = [random.choice(ex['labels']) + ' ' + self.END for ex in exs]
            parsed = [self.parse(y) for y in labels]
            max_y_len = max(len(y) for y in parsed)
            ys = torch.LongTensor(batchsize, max_y_len).fill_(0)
            for i, y in enumerate(parsed):
                for j, idx in enumerate(y):
                    ys[i][j] = idx
            if self.use_cuda:
                ys = ys.cuda(async=True)
            ys = Variable(ys)
        return xs, ys, valid_inds

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, valid_inds = self.batchify(observations)

        if len(xs) == 0:
            # no valid examples, just return the empty responses we set up
            return batch_reply

        # produce predictions either way, but use the targets if available
        predictions = self.predict(xs, ys)

        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a time
            batch_reply[valid_inds[i]]['text'] = ' '.join(
                c for c in predictions[i] if c != self.END)

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            model = {}
            model['lt'] = self.lt.state_dict()
            model['encoder'] = self.encoder.state_dict()
            model['decoder'] = self.decoder.state_dict()
            model['h2o'] = self.h2o.state_dict()
            model['longest_label'] = self.longest_label
            model['opt'] = self.opt

            with open(path, 'wb') as write:
                torch.save(model, write)

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            model = torch.load(read)

        return model['opt'], model

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.lt.load_state_dict(states['lt'])
        self.encoder.load_state_dict(states['encoder'])
        self.decoder.load_state_dict(states['decoder'])
        self.h2o.load_state_dict(states['h2o'])
        self.longest_label = states['longest_label']

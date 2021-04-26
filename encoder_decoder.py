import pandas as pd
import numpy as np
import random
import time
import argparse
import json
import array

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim



seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

class Encoder(nn.Module):
    ''' Encodes time-serise sequence '''
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are 2 stacked LSTMs)
        '''
        
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        
        # define LSTM layer
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers)
    
    def forward(self, X):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        rnn_out, self.hidden = self.rnn(X)
        return rnn_out, self.hidden
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    
class Decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, output_dim, hidden_dim, num_layers, dropout, use_bn):
        
        '''
        : param output_dim:     the number of features in the input X
        : param hidden_dim:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(Decoder, self).__init__()
        self.output_dim = output_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        
        self.rnn = nn.GRU(self.output_dim, self.hidden_dim, self.num_layers)
        self.fc_out = self.regressor()
        
    def forward(self, x_input, encoder_hidden_states):
               
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        rnn_out, self.hidden = self.rnn(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.fc_out(rnn_out.squeeze(0)) 
        
        return output, self.hidden
    
    def regressor(self):

        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.Dropout(self.dropout))
        
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim))
        regressor = nn.Sequential(*layers)

        return regressor

class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, device, target_len, training_prediction, teacher_forcing_ratio):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.target_len = target_len
        self.training_prediction = training_prediction
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    
    def forward(self, X, y):
            
            if y is None:
                self.training_prediction = 'recursive'
                
                
            # initialize
            outputs = torch.zeros(self.target_len, X.shape[1], self.decoder.output_dim).to(self.device)
            encoder_hidden = self.encoder.init_hidden(X.shape[1])
            
            # encoder process
            encoder_output, encoder_hidden = self.encoder(X)            

            #decoder process
            decoder_input = X[-1, :, :self.decoder.output_dim]  # last X sequence, shape = [batch_size, n_features]
            decoder_hidden = encoder_hidden  
            
            # ======================================================================================#
            if self.training_prediction == 'recursive':
                # predict recursively
                for t in range(self.target_len): 
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output 
                    decoder_input = decoder_output                    
           # ======================================================================================#
            if self.training_prediction == 'teacher_forcing':
                # use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len): 
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = y[t, :, :self.decoder.output_dim]
                        
                # predict recursively 
                else:
                    for t in range(self.target_len): 
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = decoder_output
            # ======================================================================================#
            if self.training_prediction == 'mixed_teacher_forcing':
                # predict using mixed teacher forcing
                for t in range(self.target_len):
                    #print("output {} space".format(t+1))
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output 
                                    
                    # predict with teacher forcing
                    if random.random() < self.teacher_forcing_ratio:
                        decoder_input = y[t, :, :self.decoder.output_dim]
                        
                        # decoder_input = y[t, :args.output_dim].unsqueeze(0) # [7day, output_dim] to [1day, output_dim]

                    # predict recursively 
                    else:
                        decoder_input = decoder_output
            # ======================================================================================#      
            return outputs


def train(model, partition, optimizer, loss_fn, args):
    ''' model training '''
    
   
    # data load
    trainloader = DataLoader(partition['train'], batch_size = args.batch_size, shuffle = True, drop_last = True)
    
    # model's mode setting
    model.train()

    # model initialization
    model.zero_grad()
    train_loss = 0.0
    
    # train
    for i, (X, y) in enumerate(trainloader):
        
        X = X.transpose(0, 1).float().to(args.device)
        y_true = y[:,:,:args.output_dim].transpose(0, 1).float().to(args.device)

        # zero the gradient
        optimizer.zero_grad()
        
        # en-decoder outputs tensor 
        y_pred = model(X, y_true)

        # compute the loss 
        loss = loss_fn(y_true, y_pred)
        
        #backpropagation
        loss.backward()
        optimizer.step()
        
        # get the batch loss
        train_loss += loss.item()
        
    train_loss = train_loss / len(trainloader)
    return model, train_loss


def validate(model, partition, loss_fn, args):
    ''' model validate '''
    
    # data load
    valloader = DataLoader(partition['val'], batch_size = args.batch_size, shuffle = False, drop_last = True)
    
    # model's mode setting
    model.eval()
    val_loss = 0.0
    
    # evaluate
    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):
            
            X = X.transpose(0, 1).float().to(args.device)
            y_true = y[:,:,:args.output_dim].transpose(0, 1).float().to(args.device)

            # en-decoder outputs tensor 
            y_pred = model(X, None)
            
            # compute the loss 
            loss = loss_fn(y_true, y_pred)

            # get the batch loss
            val_loss += loss.item()
            
    val_loss = val_loss / len(valloader)
 
    return val_loss


def test(model, partition, scaler, args):
    ''' model test '''
    
    # data load
    testloader = DataLoader(partition['test'], batch_size = 1, shuffle = False, drop_last = False)
    
    # model's mode setting
    model.eval()
    
    test_mae = 0.0
    score_list = list()
    item_loss_list = list()
    
    # evaluate
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            
            X = X.transpose(0, 1).float().to(args.device)            
            y_true = y[:,:,:args.output_dim].transpose(0, 1).float().to(args.device)
            
            
            # en-decoder outputs tensor 
            y_pred = model(X, None)
            
            y_true = y_true.view(-1, args.output_dim)
            y_pred = y_pred.view(-1, args.output_dim)
            
            # y values to cpu
            y_true = y_true.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()

            # inverse trasform y valuse 
            y_true = scaler.inverse_transform(y_true).round()
            y_pred = scaler.inverse_transform(y_pred).round()
            
            # print(y_true, end = '\n')
            # print(y_pred, end='\n')
            

            # get the batch loss
            test_mae += mean_absolute_error(y_true, y_pred)
            score = r2_score(y_pred = y_pred.transpose(), y_true = y_true.transpose(), multioutput = 'uniform_average')
            score_list.append(score)
            item_loss_list.append(abs(y_true - y_pred))
            
            
    test_mae /= len(testloader)
    score /= len(testloader)
    
    return test_mae, score_list, item_loss_list


def experiment(partition, scaler, args):

    # Encoder
    enc = Encoder(args.input_dim, args.hidden_dim, args.num_layers)
    
    # Decoder
    dec = Decoder(args.output_dim, args.hidden_dim, args.num_layers, args.dropout, args.use_bn)
    
    # Seq2Seq model
    model = Seq2Seq(enc, dec, args.device, args.target_len, args.training_prediction, args.teacher_forcing_ratio)
    model.to(args.device)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.L2_rate)
    
    # epoch-wise loss
    train_losses = []
    val_losses = []

    for epoch in range(args.num_epoch):
        
        start_time = time.time()

        model, train_loss = train(model, partition, optimizer, loss_fn, args)
        val_loss = validate(model, partition, loss_fn, args)
        
        end_time = time.time()
        
        # add epoch loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print('Epoch {},Loss(train/val) {:.3f}/{:.3f}. Took {:.2f} sec'.format(epoch+1, train_loss, val_loss, end_time-start_time))
    
    # test part
    test_mae, score_list, item_loss_list = test(model, partition, scaler, args)
    
    # ======= Add Result to Dictionary ======= #
    result = {}
    
    result['train_losses'] = train_losses #epoch 수에 의존
    result['val_losses'] = val_losses 
    
    result['test_mae'] = test_mae.round(3).item()
    
    result['r2'] = np.array(score_list).mean().round(3)
    
    item_loss = np.array(item_loss_list).mean(axis=0).mean(axis=0).astype(int)
    item_loss = list([int(x) for x in item_loss])
    
    result['item_loss'] = item_loss
     
    return vars(args), result
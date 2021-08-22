import torch
from torch import nn
from src.measures import accuracy_per_class, f1_score_function
from tqdm import tqdm


class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, embedding_matrix):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=embedding_matrix)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        text_lengths = torch.tensor([85 for _ in range(0, text.shape[0])]).float()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded.float())
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs

    def train_lstm(self, iterator, optimizer, criterion, device):
        model = self
        #initialize every epoch 
        epoch_loss = 0
        epoch_acc = 0
        #set the model in training phase
        model.train()  
        for batch in iterator:
            #resets the gradients after every batch
            optimizer.zero_grad()   
            batch = tuple(b.to(device) for b in batch)

            #retrieve text and no. of words
            text, text_lengths = batch[0], batch[0].shape[1]
            
            #convert to 1D tensor
            predictions = model(text, text_lengths).squeeze()

            #compute the loss
            loss = criterion(predictions, batch[1])        

            #compute the binary accuracy
            acc = accuracy_per_class(predictions, batch[1])   
            
            #backpropage the loss and compute the gradients
            loss.backward()       
            
            #update the weights
            optimizer.step()      
            
            #loss and accuracy
            epoch_loss += loss.item()  
            epoch_acc += acc 
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
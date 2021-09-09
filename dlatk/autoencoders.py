import torch
import torch.nn as nn

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleAE(nn.Module):
    def __init__(self, inp_dim, n_components_, layers, dropout_prob):
        super(SimpleAE, self).__init__()
        self.n_components_=n_components_
        self.layers=layers
        self.dropout=0.3 if (dropout_prob>1 or dropout_prob<0) else dropout_prob

        self.enc = []
        self.dec = []

        delta_nodes = (n_components_ - inp_dim)

        for layer in range(1,self.layers+1):
            linear_layer = nn.Linear(inp_dim+int(delta_nodes*(layer-1)/layers), \
                                    inp_dim+int(delta_nodes*(layer)/layers)
                                    )
            self.enc.append(linear_layer)
            linear_layer = nn.Linear(inp_dim+int(delta_nodes*(layer)/layers), \
                                    inp_dim+int(delta_nodes*(layer-1)/layers)
                                    )
            self.dec.append(linear_layer)            


        self.dec.reverse()
        self.enc = nn.ModuleList(self.enc)
        self.dec = nn.ModuleList(self.dec)

    def forward(self, X, training):

        enc_op = X
        for layer in self.enc:
            enc_op = layer(enc_op)
            enc_op = nn.functional.relu(enc_op)
            enc_op = nn.functional.dropout(enc_op, p=self.dropout, training=training)

        dec_op = enc_op
        for layer in self.dec[:-1]:
            dec_op = layer(dec_op)
            dec_op = nn.functional.relu(dec_op)
            dec_op = nn.functional.dropout(dec_op, p=self.dropout, training=training)
        
        dec_op = self.dec[-1](dec_op)


        return (enc_op, dec_op)
    
class AE(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=128, layers=1, epochs=3, dropout_prob=0, batch_size=32):
        self.n_components = n_components
        self.n_components_ = n_components 
        self.layers = layers
        self.epochs = epochs
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.ae = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.optim = None
        self.loss_fn = nn.MSELoss()
        
    def fit(self, X, y=None):
        X = self.scaler.fit_transform(X)
        X_train, X_test = train_test_split(X, train_size=0.9, random_state=42)
        train_epoch_losses = []
        val_epoch_losses = [-np.inf]
        e=0
        self.ae = SimpleAE(X.shape[1], self.n_components, self.layers, self.dropout_prob).to(self.device)
        #Need to look into lr and decay params
        self.optim = torch.optim.AdamW(self.ae.parameters(), lr=1e-3)
        #for e in range(self.epochs):
        while ((e < 2) or (not (val_epoch_losses[-1] > val_epoch_losses[-2] > val_epoch_losses[-3]))):
            train_epoch_loss = []
            val_epoch_loss = []
            self.ae.train()
            for i in range(0, len(X_train), self.batch_size):
                self.optim.zero_grad()
                inp = torch.FloatTensor(X_train[i:i+self.batch_size]).to(self.device)
                (_, dec) = self.ae(inp, True)
                loss = self.loss_fn(dec, inp)
                del dec, inp
                loss.backward()
                train_epoch_loss.append(loss.detach().cpu().numpy())
                self.optim.step()
            
            self.ae.eval()
            for i in range(0, len(X_test), self.batch_size):
                inp = torch.FloatTensor(X_test[i:i+self.batch_size]).to(self.device)
                with torch.no_grad():
                    (_, dec) = self.ae(inp, False)
                loss = self.loss_fn(dec, inp)
                del dec, inp
                val_epoch_loss.append(loss.detach().cpu().numpy())

            train_epoch_losses.append(np.mean(train_epoch_loss))
            val_epoch_losses.append(np.mean(val_epoch_loss))
            print ("Mean Train Loss for epoch %d: %f"%(e+1, train_epoch_losses[-1]))
            print ("Mean Val Loss for epoch %d: %f"%(e+1, val_epoch_losses[-1]))
            e+=1
        return self

    def transform(self, X, y=None):
        self.ae.eval()
        X = self.scaler.transform(X)
        X_reduced = []
        for i in range(0, len(X), self.batch_size):
            inp = torch.FloatTensor(X[i: i+self.batch_size]).to(self.device)
            with torch.no_grad():
                (enc, _) =  self.ae(inp, False)
                X_reduced.append(enc.detach().cpu().numpy())
                del enc
            
        X_reduced = np.concatenate(X_reduced, axis=0)
        return X_reduced
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

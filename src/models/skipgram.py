# Adapted from: https://github.com/fanglanting/skip-gram-pytorch/tree/master

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    """Skip-gram model."""

    def initialize_embeddings(self):
        initialization_range = 0.5 / self.embedding_dimension
        self.u_embeddings.weight.data.uniform_(-initialization_range, initialization_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def __init__(
            self,
            vocabulary_size: int,
            embedding_dimension: int,
    ):
        super(SkipGram, self).__init__()
        self.u_embeddings = nn.Embedding(vocabulary_size, embedding_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocabulary_size, embedding_dimension, sparse=True)
        self.embedding_dimension = embedding_dimension
        
        self.initialize_embeddings()

    def forward(self, u_positive, v_positive, v_negative, batch_size):
        # Compute embeddings
        u_positive_embeddings = self.u_embeddings(u_positive)
        v_positive_embeddings = self.v_embeddings(v_positive)
        v_negative_embeddings = self.v_embeddings(v_negative)

        # Get positive score
        positive_score = torch.mul(u_positive_embeddings, v_positive_embeddings)
        positive_score = torch.sum(positive_score, dim=1)
        positive_target = F.logsigmoid(positive_score).squeeze()

        # Get negative score
        negative_score = torch.bmm(v_negative_embeddings, u_positive_embeddings.unsqueeze(2)).squeeze()
        negative_score = torch.sum(negative_score, dim=1)
        negative_target = F.logsigmoid(-negative_score).squeeze() # Negative samples have negative labels

        # Compute loss as the sum of positive and negative log likelihoods
        loss = positive_target + negative_target

        return -loss.sum() / batch_size
    
    def get_input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()
    
    def save_input_embeddings(self, path):
        torch.save(self.get_input_embeddings(), path)
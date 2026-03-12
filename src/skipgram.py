import numpy as np
import pickle
from src.negative_sampling import sample_negative_words
import random


class SkipGramModel:
    def __init__(self,vocab_size,embedding_dim):
        
        self.W1 = np.random.randn(vocab_size,embedding_dim) * 0.01
        self.W2 = np.random.randn(vocab_size,embedding_dim) * 0.01
        
        
        
    def sigmoid(self,x):
        # Clips the values to avoid "Overflow" errors with exp
        x = np.clip(x,-15,15)
        
        return 1 / (1 + np.exp(-x))
    
    
    
    def forward(self, target_idx, context_idx):
        # 1. Grab the specific 100-number vectors (profiles)
        v_target = self.W1[target_idx]
        
        v_context = self.W2[context_idx]
        
        # 2. Compute the dot product score
        # This tells us how similar the model currently thinks they are
        score = np.dot(v_target,v_context)
        
        #3. GEt the probability
        prediction = self.sigmoid(score)
        
        return prediction, v_target,v_context
    
    


def train_step(model,target_idx, pos_context_idx,neg_context_indices,learning_rate):
        
    #prediction(y_hat)
    pred_pos ,v_target,v_context = model.forward(target_idx,pos_context_idx)
    
    
    
    # Error (y_hat -y) -> y is 1 for positive
    error_pos = pred_pos - 1
    
    
    #Los for positive pair: -log(pred_pos)
    step_loss = -np.log(pred_pos + 1e-10)
    
    #calculate gradients
    grad_target = error_pos * v_context
    grad_context  = error_pos * v_target
    
    
    #update w1 and w2 for this positive pair
   # model.W1[target_idx] -= learning_rate * grad_target
    model.W2[pos_context_idx] -= learning_rate * grad_context
    
    total_grad_target = grad_target
   
    
    for neg_idx in neg_context_indices:
        pred_neg, v_target, v_neg = model.forward(target_idx, neg_idx)
        
        #Error(y_hat -y ) -> y is 0 for negative
        err_neg = pred_neg - 0
        
        #Loss for negative pair: -log(1-pred_neg)
        step_loss += -np.log(1-pred_neg + 1e-10)
        
        
        #calculate gradients
        grad_target_neg = err_neg * v_neg
        grad_neg = err_neg * v_target
        
        total_grad_target+= grad_target_neg
        
        
        #update W1 and W2  for this negative pair
       
        model.W2[neg_idx] -= learning_rate * grad_neg
        
    model.W1[target_idx] -= learning_rate * total_grad_target
    return step_loss
        


def generate_training_pairs(corpus,window_size):
    for i in range(len(corpus)):
        target = corpus[i]
        
        #identify the boundaries of our window
        start = max(0,i-window_size)
        end = min(len(corpus),i+window_size+1)
        
        for j in range(start,end):
            if i == j:
                continue
            context = corpus[j]
            
            yield (target,context)
            
            
            
            
            
            
            
            
            
            
            
            
def train_model(model, corpus, unigram_table, epochs, window_size,initial_lr):
    
    
    learning_rate = initial_lr
    
    for epoch in range(epochs):
        
        #1. shuffle the reviews to avoid order bias.
        random.shuffle(corpus)
        
        
        total_loss = 0
        
        pairs_processed = 0
        
        
        for review in corpus: # Loop through each review
            for target_idx, context_idx in generate_training_pairs(review, window_size):
                #1. Get 5 negative samples from the bucket
                neg_indices = sample_negative_words(unigram_table,num_samples=5)
                
                
                #2. Running train step function
                # Inside your loop
                loss = train_step(model, target_idx, context_idx, neg_indices, learning_rate)
                total_loss += loss
                
                pairs_processed +=1
                
                
                if pairs_processed % 10000 == 0:
                    print(f"Epoch {epoch+1} | Pairs: {pairs_processed} | LR: {learning_rate:.5f} | Loss: {(total_loss/pairs_processed):.5f}")
                
         

            if epoch % 3 == 0:
                learning_rate *= 0.9      
        
        print(f"End of epoch{epoch + 1}.New LR: {learning_rate:5f}")
        
        
    save_embeddings(model.W1,"./../data/embeddings.npy")
    print("Training complete. Embeddings saved!")
    
    
def save_embeddings(matrix,filename):
    np.save(filename,matrix) 
                
            
            
            
            
            
            
            

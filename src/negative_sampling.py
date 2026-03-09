import numpy as np
from collections import Counter


def build_unigram_table(corpus, vocab_size, table_size=100_000_000):
    
    # count how many times each word ID appears in the corpus
    counts = Counter(corpus)
    
    #create an array of counts for every ID in our 10k word vocab
    #raw_counts = np.array([counts.get(i,0) + 1e-6 for i in range(vocab_size)])
    
    
    raw_counts_list = []
    
    
    for i in range(vocab_size):
        
        
        count_from_data = counts.get(i,0)
        
        final_count = count_from_data + 0.0000001
        
        raw_counts_list.append(final_count)
        
    raw_counts = np.array(raw_counts_list)
    
    
    
    #appy the squishing
    pow_counts = np.power(raw_counts,0.75)
    
    #turn the probabilites to sum to 1
    probabilities = pow_counts / np.sum(pow_counts)
    
    
    #prefill a giant table with id based on probabilities
    table = np.random.choice(range(vocab_size),size=table_size,p=probabilities)
    
    return table


def sample_negative_words(unigram_table,num_samples):
    # to get 'n' fakew words,
    indices = np.random.randint(low=0,high=len(unigram_table),size=num_samples)
    
    return unigram_table[indices]
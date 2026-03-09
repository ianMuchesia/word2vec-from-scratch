import pandas as pd
from src.tokenizer import Tokenizer


df = pd.read_csv('./data/IMDB Dataset.csv')


#print(df.head())


tok = Tokenizer()

reviews = df['review'].tolist()

# for review in df['review']:
#     reviews.append(review.replace('\n',''))
    
    
# print(reviews[:20])

tok.build_vocab(reviews)


encoded_reviews = tok.encode(reviews)


flattened_corpus = []


for review_ids in encoded_reviews:
    flattened_corpus.extend(review_ids)
    
    
corpus_string = " ".join(map(str,flattened_corpus))


with open("./data/corpus.txt", 'w') as f:
    f.write(corpus_string)

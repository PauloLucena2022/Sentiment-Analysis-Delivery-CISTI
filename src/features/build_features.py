import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True)

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile("["
                               u"\\U0001F600-\\U0001F64F"
                               u"\\U0001F300-\\U0001F5FF"
                               u"\\U0001F600-\\U0001F6FF"
                               u"\\U0001F700-\\U0001F77F"
                               u"\\U0001F780-\\U0001F7FF"
                               u"\\U0001F800-\\U0001F8FF"
                               u"\\U0001F900-\\U0001F9FF"
                               u"\\U0001FA00-\\U0001FA6F"
                               u"\\U0001FA70-\\U0001FAFF"
                               u"\\U0001F004"
                               u"\\U0001F0CF"
                               u"\\U0001F170-\\U0001F251"
                               u"\\U00002702-\\U00002780"
                               u"\\U000024C2-\\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', str(text))
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def remove_links(text: str) -> str:
    return re.sub(r'www\S+|http\S+|ftp\S+', '', text)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def process_text_pipeline(text: str, stop_words: set, stemmer: RSLPStemmer) -> str:
    text = remove_emojis(text)
    text = remove_links(text)

    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens] # Tokenização and lowercase

    tokens = [token for token in tokens if token not in string.punctuation] # punctuation removed

    tokens = [token for token in tokens if token not in stop_words] # Stop words removed

    stemmed_tokens = [stemmer.stem(token) for token in tokens] # Stemming
    
    return ' '.join(stemmed_tokens)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def to_sentiment(score: int) -> str:
    return "Positive" if int(score) >= 4 else "Negative"
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def build_model_features(df: pd.DataFrame, text_column: str = 'content', score_column: str = 'score') -> pd.DataFrame:
    df_processed = df.copy()
    
    print("Initiate text preprocessing...")
    
    stop_words_pt = set(stopwords.words('portuguese'))
    stemmer_pt = RSLPStemmer()
    
    df_processed['stemmed_text'] = df_processed[text_column].apply(
        lambda x: process_text_pipeline(x, stop_words_pt, stemmer_pt)
    )
    
    df_processed['sentiment'] = df_processed[score_column].apply(to_sentiment)
    
    print("Preprocessing completed!")
    
    return df_processed[['reviewId', text_column, 'stemmed_text', score_column, 'sentiment']]
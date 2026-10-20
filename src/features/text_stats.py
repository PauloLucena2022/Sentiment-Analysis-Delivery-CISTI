import pandas as pd 
import string 
from collections import Counter
import nltk 
from nltk.corpus import stopwords 

def add_word_count(df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
    df = df.copy()
    df['word_count'] = df[text_column].dropna().astype(str).str.split().apply(len)

    min_w = df['word_count'].min()
    max_w = df['word_count'].max()
    mean_w = df['word_count'].mean()
    single_word_count = len(df[df['word_count'] == 1])

    print(f"=== Estatísticas de Palavras ({text_column}) ===")
    print(f"Mínimo de palavras: {min_w}")
    print(f"Máximo de palavras: {max_w}")
    print(f"Média de palavras: {mean_w:.2f}")
    print(f"Avaliações com apenas 1 palavra: {single_word_count}")
    print("============================================\n")

    return df
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def get_most_frequent_words(df: pd.DataFrame, text_column: str = 'content', top_n: int = 20) -> pd.DataFrame:
    nltk.download('stopwords', quiet = True)
    stop_words = set(stopwords.words('portuguese'))

    all_text = ''.join(df[text_column].dropna().astype(str)).lower()
    translator = str.maketrans('','', string.punctuation)
    clean_text = all_text.translate(translator)

    words = [w for w in clean_text.split() if w not in stop_words and len (w) > 2]
    most_common = Counter(words).most_common(top_n)

    return pd.DataFrame(most_common, columns = ['word', 'frequency'])

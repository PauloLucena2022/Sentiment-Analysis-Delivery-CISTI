import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, VotingClassifier

def prepare_training_data(df: pd.DataFrame, text_col: str = 'stemmed_text', target_col: str = 'sentiment'):
    print("Split data in Train and Test")
    
    X = df[text_col]
    y = df[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_encoded, 
        test_size = 0.3, 
        stratify = y_encoded, 
        random_state = 42
    )
    
    print("Initiate Vectorizer (Bag of Words)...")
    
    bag_of_words = CountVectorizer()
    
    X_train_num = bag_of_words.fit_transform(X_train)
    X_test_num = bag_of_words.transform(X_test)
    
    joblib.dump(bag_of_words, 'models/bag_of_words_vectorizer.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Vetorizer and Encoder saves in 'models/'.")
    
    return X_train_num, X_test_num, y_train, y_test
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Training Bagging functions
def train_bagging_knn(X_train, y_train):
    print("Training Bagging - KNN...")
    knn_base = KNeighborsClassifier(n_neighbors = 17, metric = 'cosine', n_jobs = -1, weights = 'distance')
    bagging_knn = BaggingClassifier(
        estimator = knn_base,
        n_estimators = 10,
        max_samples = 0.8,
        bootstrap = True,
        bootstrap_features = True,
        n_jobs = -1
    )
    bagging_knn.fit(X_train, y_train)
    joblib.dump(bagging_knn, 'models/bagging_knn.pkl')
    return bagging_knn 

def train_bagging_mlp(X_train, y_train):
    print("Training Bagging - MLP...")
    mlp_base = MLPClassifier(
        hidden_layer_sizes = (4, 8),
        activation = 'relu',
        solver ='adam',
        max_iter = 16,
        batch_size = 'auto',
        learning_rate_init =0.0001
    )
    bagging_mlp = BaggingClassifier(
        estimator = mlp_base,
        n_estimators = 10,
        max_samples = 0.8,
        bootstrap = True,
        bootstrap_features = True,
        n_jobs = -1
    )
    bagging_mlp.fit(X_train, y_train)
    joblib.dump(bagging_mlp, 'models/bagging_mlp.pkl')
    return bagging_mlp


def train_bagging_svm(X_train, y_train):
    print("Training Bagging - SVM...")
    svm_base = SVC(kernel = 'linear', C = 1.0, gamma = 'scale')
    bagging_svm = BaggingClassifier(
        estimator = svm_base,
        n_estimators = 10,
        max_samples = 0.8,
        bootstrap = True,
        bootstrap_features = True,
        n_jobs = -1
    )
    bagging_svm.fit(X_train, y_train)
    joblib.dump(bagging_svm, 'models/bagging_svm.pkl')
    return bagging_svm
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Training Voting function
def train_hard_voting(X_train, y_train):
    print("Training Hard Voting (MLP, SVM, KNN)...")
    mlp_voting = MLPClassifier(
        hidden_layer_sizes = (6, 2),
        activation = 'relu',
        solver = 'adam',
        batch_size = 'auto',
        max_iter = 2,
        learning_rate_init = 0.0001
    )

    knn_voting = KNeighborsClassifier(n_neighbors = 6, metric = 'cosine', weights = 'distance', n_jobs = -1)
    svm_voting = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
    
    model_hard_voting = VotingClassifier(
        estimators=[
            ('mlp', mlp_voting),
            ('svm', svm_voting),
            ('knn', knn_voting)
        ],
        voting='hard'
    )
    model_hard_voting.fit(X_train, y_train)
    joblib.dump(model_hard_voting, 'models/hard_voting.pkl')
    return model_hard_voting



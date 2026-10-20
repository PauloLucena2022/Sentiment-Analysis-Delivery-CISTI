import os 
from src.data.load_data import load_dataset
from src.features.build_features import build_model_features
from src.models.train_model import (
    prepare_training_data,
    train_bagging_knn,
    train_bagging_svm,
    train_bagging_mlp,
    train_hard_voting
)
from src.models.evaluate_model import evaluate_classifier

def main():
    print("ANALYSIS SENTIMENT PIPELINE - DELIVERY FOOD")

    os.makedirs('models', exist_ok = True)

    # 1. Loading Dataset
    file_path = 'data/raw/delivery_alimentos.csv'
    print(f'\nLoading dataset: {file_path}')
    df = load_dataset(file_path)

    # 2. Preprocessing
    print('\nInitiate process text and features extraction')
    df_processed = build_model_features(df, text_column = 'content', score_column = 'score')

    # 3. Vectorizer and Partitioning (Bag of Words)
    print('\nPartitioning and Vectorizer')
    X_train_num, X_test_num, y_train, y_test = prepare_training_data(df = df_processed, text_col = 'stemmed_text', target_col = 'sentiment')

    # 4. Training Ensemble Models
    print('\nTraining Ensemble classificators')
    bagging_knn_model = train_bagging_knn(X_train_num, y_train)
    bagging_svm_model = train_bagging_svm(X_train_num, y_train)
    bagging_mlp_model = train_bagging_mlp(X_train_num, y_train)
    hard_voting_model = train_hard_voting(X_train_num, y_train)

    # 5. Evaluation
    print('\nEvaluating ensemble models accuracy')

    evaluate_classifier(bagging_knn_model, X_test_num, y_test, 'Bagging - KNN')
    evaluate_classifier(bagging_svm_model, X_test_num, y_test, 'Bagging - SVM')
    evaluate_classifier(bagging_mlp_model, X_test_num, y_test, 'Bagging - MLP')
    evaluate_classifier(hard_voting_model, X_test_num, y_test, 'Hard Voting - KNN/SVM/MLP')

    print('\nPIPELINE COMPLETED!')

if __name__ == '__main__':
    main()
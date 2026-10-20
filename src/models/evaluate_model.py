from sklearn.metrics import accuracy_score

def evaluate_classifier(model, X_test, y_test, model_name = "Model"):
    print(f"\n Evaluation: {model_name}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred * 100)
    print(f"Accuracy: {acc:.4f}%")

    return acc
def accuracy_score(y_test, y_pred):
    pred_true_count = 0
    for true_label, pred in zip(y_test, y_pred):
        if true_label == pred:
            pred_true_count += 1
    return pred_true_count / len(y_test)

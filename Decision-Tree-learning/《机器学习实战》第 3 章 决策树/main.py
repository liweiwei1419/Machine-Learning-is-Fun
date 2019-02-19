import pandas as pd

from tree import DecisionTreeClassifier
from metrics import accuracy_score
from utils import train_test_split

if __name__ == '__main__':
    column_names = ['parents', 'has_nurs', 'form', 'children',
                    'housing', 'finance', 'social', 'health', 'classes']
    data = pd.read_csv('./nursery.data', names=column_names)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=666)

    dt = DecisionTreeClassifier()

    dt.fit(X, y)

    y_pred = dt.predict(X_test)
    print(y_pred)
    print()

    score = accuracy_score(y_test, y_pred)
    print(score)

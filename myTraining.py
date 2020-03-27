import pandas as pd
import numpy as np
import pickle


def spilt(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indicies = shuffled[:test_set_size]
    train_indicies = shuffled[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    train, test = spilt(df, 0.2)
    X_train = train[['fever', 'bodyPain', 'age',
                     'runningNose', 'diffBreathing']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age',
                   'runningNose', 'diffBreathing']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(2400,)
    Y_test = test[['infectionProb']].to_numpy().reshape(599,)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # close the file
    file.close()

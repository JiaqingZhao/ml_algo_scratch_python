import decision_tree as dt
from utils import *

class random_forest_classifier():

    def __init__(self, n_features, n_trees = 200, max_depth = 10, min_size = 10,  n_sample_rate= .9):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.n_sample_rate = n_sample_rate
        self.grown_forest = None

    def build_model(self, data):
        trees = []
        for i in range(self.n_trees):
            n_samples = get_random_subset(round(self.n_sample_rate * len(data)), len(data))
            t = dt.tree(self.max_depth, self.min_size, self.n_features)
            t.fit([data[i] for i in n_samples])
            trees.append(t.grown_tree)
        self.grown_forest = trees

    def predict(self,newdata):
        predictions = []
        for row in newdata:
            pred_per_row = []
            for single_tree in self.grown_forest:
                pred_per_row.append(dt.tree(self.max_depth, self.min_size, self.n_features).predict_with_single_tree(single_tree, row))
            predictions.append(max(set(pred_per_row), key=pred_per_row.count))
        return predictions


if __name__ == '__main__':

    import pandas as pd
    import utils

    data = pd.read_csv("sample_data\sonar.csv").values.tolist()


    random.shuffle(data)
    n_train_data = int(len(data) * .9)
    train, test = data[: n_train_data], data[n_train_data:]


    rfc = random_forest_classifier(n_features = 7, n_trees = 200, max_depth = 10, min_size = 1,  n_sample_rate= .9)
    rfc.build_model(train)


    predictions = rfc.predict(test)
    labels = [i[-1] for i in test]

    n_corr = 0
    for i,j in zip(labels, predictions):
        if i == j:
            n_corr += 1
    print(n_corr / len(labels))


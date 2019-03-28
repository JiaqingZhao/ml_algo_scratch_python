import random
from utils import *

class node():
    """
    The node class performs all the functionalities that would happen during a node formation/split.
    The general idea is that when a data blob get passed into the node class, we can
    1. try to get the best split of it and return two smaller blobs
    2. try to figure out the predominant label if the node is a terminal/leaf node
    3. automatically detect if it is a terminal node under three circumstances a) inseperable b) reached maximum depth
        and c) reached minimum size
    """
    def __init__(self, data, n_features, is_leaf = False):
        self.data = data # input data, will be deleted when the seperation takes place
        self.n_features = n_features
        self.split_value = None #
        self.data_left = None # left data
        self.data_right = None # right data
        self.left_node = None
        self.right_node = None
        self.split_position = None # (a,b) a tuple for splitting position
        self.is_leaf = is_leaf # is the current node a leaf?
        self.label = None # if so what is the label for this node?
        self.catagories = [row[-1] for row in data] # catagories


    def get_gini_index(self,left,right):
        gini_index = 0
        for data_partial in left,right:
            if len(data_partial) == 0:
                continue
            score = 0
            for category in set(self.catagories):
                p = [row[-1] for row in data_partial].count(category) / len(data_partial)
                score += p * p
            gini_index += (1 - score) * (len(data_partial) / len(self.data))
        return gini_index


    def split(self, data, r, c):
        split_value = data[r][c]
        left, right = [], []
        for row in data:
            if row[c] <= split_value:
                left.append(row)
            else:
                right.append(row)
        return left, right,split_value

    def get_split_point(self,min_size,depth,max_depth):
        if len(self.data) < min_size:
            self.get_leaf_label()
            return
        n_total_features = len(self.data[0]) - 1 # presumbaly we are attaching labels to the last column
        features = get_random_subset(self.n_features, n_total_features)
        r, c, gini_index = None, None, 999
        for row_index in range(len(self.data)):
            for feature in features:
                #print(row_index,feature,gini_index)
                left, right,split_value = self.split(self.data, row_index, feature)
                current_gini_index = self.get_gini_index(left, right)
                if current_gini_index < gini_index:
                    r,c, gini_index = row_index, feature, current_gini_index
                    self.data_left = left
                    self.data_right = right
                    self.split_value = split_value
                    self.split_position = (r,c)

        if len(self.data_left) == 0 or len(self.data_right) == 0 or depth >= max_depth:
            self.get_leaf_label()

        self.data = None

    def get_leaf_label(self):
        self.is_leaf = True
        self.label = max(set(self.catagories), key=self.catagories.count)


class tree():
    """
    The tree class does the following:
    1. grow a tree
    2. keep track of depth, min_size and leaf nodes to keep trees slim
    3. provide predictions from a grown tree
    """

    def __init__(self,  max_depth, min_size, n_features, max_leaf_node = False):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.leaf_node_total = 0
        self.max_leaf_node = max_leaf_node
        self.grown_tree = None

    def build_tree(self, cur_level_data, depth = 0):
        if self.max_leaf_node == False:

            pass
        else:

            if self.leaf_node_total >= self.max_leaf_node:
                return
        n = node(cur_level_data,self.n_features)
        n.get_split_point(self.min_size,depth,self.max_depth)
        left_group = n.data_left
        right_group = n.data_right
        n.data_left, n.data_right = None, None
        if n.is_leaf == False:
            n.left_node = self.build_tree(left_group, depth + 1)
            n.right_node = self.build_tree(right_group, depth + 1)
        else:
            self.leaf_node_total += 1
        return n

    def fit(self, cur_level_data):
        result = self.build_tree(cur_level_data)
        self.grown_tree = result

    def predict_with_single_tree(self, init_tree, row):

        if init_tree.label is not None:
            return init_tree.label
        r,c = init_tree.split_position
        if row[c] <= init_tree.split_value:
            return self.predict_with_single_tree(init_tree.left_node, row)
        else:
            return self.predict_with_single_tree(init_tree.right_node, row)

    def predict(self, new_data):
        predictions = []
        for row in new_data:
            prediction = self.predict_with_single_tree(self.grown_tree, row)
            predictions.append(prediction)
        return predictions

if __name__ == '__main__':

    import pandas as pd
    data = pd.read_csv("sample_data\sonar.csv").values.tolist()
    # test nodes

    t = tree(3,3,3)
    t.fit(data)
    print(t.grown_tree.left_node.split_value)
    predictions = t.predict(data)
    print(predictions)
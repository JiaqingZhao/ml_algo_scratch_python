import random

def get_random_subset(n_target, n_total):
    n_all = [i for i in range(n_total)]
    random.shuffle(n_all)
    return n_all[:n_target]

if __name__ == '__main__':
    print(get_random_subset(6,19))
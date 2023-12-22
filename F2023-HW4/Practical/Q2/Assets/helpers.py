import random

import numpy as np

class Grid():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.moves = [[[] for _ in range(m)] for _ in range(n)]

    def _get_indeces(self, cell_num):
        return cell_num // self.m, cell_num % self.m

    def add_path(self, i, j):
        """
        creates a path from cell i to j
        """
        if i == j + 1:
            i, j = self._get_indeces(i)
            self.moves[i][j].append('L')
        elif i == j - 1:
            i, j = self._get_indeces(i)
            self.moves[i][j].append('R')
        elif i == j + self.m:
            i, j = self._get_indeces(i)
            self.moves[i][j].append('U')
        elif i == j - self.m:
            i, j = self._get_indeces(i)
            self.moves[i][j].append('D')

    def has_path(self, i, j):
        """
        returns True if cells i and j are adjacent
        """
        y_1, x_1 = self._get_indeces(i)
        y_2, x_2 = self._get_indeces(j)
        if i == j + 1:
            return 'L' in self.moves[y_1][x_1] and 'R' in self.moves[y_2][x_2]
        elif i == j - 1:
            return 'R' in self.moves[y_1][x_1] and 'L' in self.moves[y_2][x_2]
        elif i == j + self.m:
            return 'U' in self.moves[y_1][x_1] and 'D' in self.moves[y_2][x_2]
        elif i == j - self.m:
            return 'D' in self.moves[y_1][x_1] and 'U' in self.moves[y_2][x_2]
        return False

    def get_neighbors(self, i):
        """
        returns cell number of neighbors for cell i
        """
        ret = []
        y, x = self._get_indeces(i)
        for move in self.moves[y][x]:
            if move == 'R':
                ret.append(i + 1)
            elif move == 'L':
                ret.append(i - 1)
            elif move == 'U':
                ret.append(i - self.m)
            elif move == 'D':
                ret.append(i + self.m)
        return ret

    def get_random_neighbour(self, i):
        return random.choice(self.get_neighbors(i))

def get_sample_grid()-> Grid:
    ret = Grid(2, 3)
    ret.moves[0][0] = ['R']
    ret.moves[0][1] = ['L', 'R', 'D']
    ret.moves[0][2] = ['L']
    ret.moves[1][0] = ['R']
    ret.moves[1][1] = ['L', 'R', 'U']
    ret.moves[1][2] = ['L']
    return ret

def PF_checker(pf_func):
    num_tests = 100
    tests_to_check = [i + 1 for i in range(num_tests)]
    correct = 0
    for idx, i in enumerate(tests_to_check):
        grid, prob, obs, actuals = get_test_case(i)
        if pf_func(grid, prob, obs) == actuals[-1]:
            correct += 1
    if correct / num_tests >= 0.7:
        print('PF PASSED!')
    else:
        print('PF FAILED!')

def consistent_log(x):
    if x < 0:
        raise Exception()
    if x == 0:
        return -np.inf
    return np.log(x)

def calculate_seq_prob(grid, prob, obs, seq):
    assert len(obs) == len(seq)
    ret = consistent_log(prob[seq[0], obs[0]])
    num_cells = grid.n * grid.m

    transition_mat = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in grid.get_neighbors(i):
            transition_mat[i, j] = 1
        transition_mat[i] = transition_mat[i] / np.sum(transition_mat[i])

    for i in range(1, len(obs)):
        ret += consistent_log(transition_mat[seq[i - 1], seq[i]]) + consistent_log(prob[seq[i], obs[i]])
    
    return ret

def get_GT_HMM(test_num = 1):
    with open(f'Assets/tests/out-HMM/{test_num}.txt') as f:
        return list(map(int, f.readline().split()))
    
def viterbi_checker(vit_func):
    num_tests = 100
    test_to_check = [i + 1 for i in range(num_tests)]
    for idx, i in enumerate(test_to_check):
        grid, prob, obs, _ = get_test_case(i)
        arr_GT = get_GT_HMM(i)
        arr_user = vit_func(grid, prob, obs)
        if calculate_seq_prob(grid, prob, obs, arr_user) < calculate_seq_prob(grid, prob, obs, arr_GT):
            print('VITERBI FAILED!')
            return
    print('VITERBI PASSED!')
        

def get_test_case(test_num=1):
    """
    input: number of test case

    output:
    [0] -> grid
    [1] -> prob matrix
    [2] -> observations
    [3] -> actual states
    """
    with open(f'Assets/tests/in/input{test_num}.txt') as f:
        n, m = map(int, f.readline().split())
        grid_ret = Grid(n, m)
        for i in range(n*m):
            for j in f.readline().split():
                grid_ret.add_path(i, int(j))
        p_ret = np.zeros((n*m, n*m))
        for i in range(n*m):
            for j, flt in enumerate(f.readline().split()):
                p_ret[i, j] = float(flt)
        actuals = [int(i) for i in f.readline().split()]
        observations = [int(i) for i in f.readline().split()]

        return grid_ret, p_ret, observations, actuals
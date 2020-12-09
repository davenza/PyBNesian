import numpy as np
import pandas as pd


def generate_normal_data(size, seed=0):
    np.random.seed(seed)

    a_array = np.random.normal(3, 0.5, size=size)
    b_array = 2.5 + 1.65*a_array + np.random.normal(0, 2, size=size)
    c_array = -4.2 - 1.2*a_array + 3.2*b_array + np.random.normal(0, 0.75, size=size)
    d_array = 1.5 - 0.9*a_array + 5.6*b_array + 0.3 * c_array + np.random.normal(0, 0.5, size=size)


    return pd.DataFrame({
                    'a': a_array,
                    'b': b_array,
                    'c': c_array,
                    'd': d_array
                    })

def generate_normal_data_indep(size, seed=0):
    np.random.seed(seed)

    a_array = np.random.normal(3, 0.5, size=size)
    b_array = np.random.normal(2.5, 2, size=size)
    c_array = -4.2 - 1.2*a_array + 3.2*b_array + np.random.normal(0, 0.75, size=size)
    d_array = 1.5 - 0.3 * c_array + np.random.normal(0, 0.5, size=size)


    return pd.DataFrame({
                    'a': a_array,
                    'b': b_array,
                    'c': c_array,
                    'd': d_array
                    })



def generate_discrete_data_uniform(size, seed=0):
    np.random.seed(seed)

    a_dict = np.asarray(['a1', 'a2'])
    b_dict = np.asarray(['b1', 'b2', 'b3'])
    c_dict = np.asarray(['c1', 'c2'])
    d_dict = np.asarray(['d1', 'd2', 'd3', 'd4'])

    return pd.DataFrame({'A': a_dict[np.random.randint(0, a_dict.size, size=size)],
                         'B': b_dict[np.random.randint(0, b_dict.size, size=size)],
                         'C': c_dict[np.random.randint(0, c_dict.size, size=size)],
                         'D': d_dict[np.random.randint(0, d_dict.size, size=size)]
                        }, dtype='category')


def generate_discrete_data_dependent(size, seed=0):
    np.random.seed(seed)

    a_dict = np.asarray(['a1', 'a2'])
    b_dict = np.asarray(['b1', 'b2', 'b3'])
    c_dict = np.asarray(['c1', 'c2'])
    d_dict = np.asarray(['d1', 'd2', 'd3', 'd4'])

    a_values = a_dict[np.random.choice(a_dict.size, size, p=[0.75, 0.25])]
    b_values = np.empty_like(a_values)
    c_values = np.empty_like(a_values)
    d_values = np.empty_like(a_values)

    a1_indices = a_values == 'a1'

    b_values[a1_indices] = b_dict[np.random.choice(b_dict.size, np.sum(a1_indices), p=[0.33, 0.33, 0.34])]
    b_values[~a1_indices] = b_dict[np.random.choice(b_dict.size, np.sum(~a1_indices), p=[0, 0.8, 0.2])]

    a1b1_indices = np.logical_and(a_values == 'a1', b_values == 'b1')
    a1b2_indices = np.logical_and(a_values == 'a1', b_values == 'b2')
    a1b3_indices = np.logical_and(a_values == 'a1', b_values == 'b3')
    a2b1_indices = np.logical_and(a_values == 'a2', b_values == 'b1')
    a2b2_indices = np.logical_and(a_values == 'a2', b_values == 'b2')
    a2b3_indices = np.logical_and(a_values == 'a2', b_values == 'b3')

    c_values[a1b1_indices] = c_dict[np.random.choice(c_dict.size, np.sum(a1b1_indices), p=[0.5, 0.5])]
    c_values[a1b2_indices] = c_dict[np.random.choice(c_dict.size, np.sum(a1b2_indices), p=[0.75, 0.25])]
    c_values[a1b3_indices] = c_dict[np.random.choice(c_dict.size, np.sum(a1b3_indices), p=[0.2, 0.8])]
    c_values[a2b1_indices] = c_dict[np.random.choice(c_dict.size, np.sum(a2b1_indices), p=[1, 0])]
    c_values[a2b2_indices] = c_dict[np.random.choice(c_dict.size, np.sum(a2b2_indices), p=[0, 1])]
    c_values[a2b3_indices] = c_dict[np.random.choice(c_dict.size, np.sum(a2b3_indices), p=[0.01, 0.99])]

    c1_indices = c_values == 'c1'
    c2_indices = c_values == 'c2'

    d_values[c1_indices] = d_dict[np.random.choice(d_dict.size, np.sum(c1_indices), p=[0.25, 0.25, 0.25, 0.25])]
    d_values[c2_indices] = d_dict[np.random.choice(d_dict.size, np.sum(c2_indices), p=[0.7, 0, 0.15, 0.15])]

    return pd.DataFrame({'A': a_values,
                         'B': b_values,
                         'C': c_values,
                         'D': d_values
                        }, dtype='category')
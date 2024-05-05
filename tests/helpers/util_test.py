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


def generate_hybrid_data(size, seed=0):
    #
    #   Generate data from:
    #
    #   A   B   C
    #    \  |  /
    #     \ | /
    #       v
    #       D
    np.random.seed(seed)

    a_dict = np.asarray(['a1', 'a2'])
    a_values = a_dict[np.random.choice(a_dict.size, size, p=[0.75, 0.25])]

    b_dict = np.asarray(['b1', 'b2', 'b3'])
    b_values = b_dict[np.random.choice(b_dict.size, size, p=[0.3, 0.4, 0.3])]

    c_values = -4.2 + np.random.normal(0, 0.75, size=size)

    a1b1_indices = np.logical_and(a_values == 'a1', b_values == 'b1')
    a1b2_indices = np.logical_and(a_values == 'a1', b_values == 'b2')
    a1b3_indices = np.logical_and(a_values == 'a1', b_values == 'b3')
    a2b1_indices = np.logical_and(a_values == 'a2', b_values == 'b1')
    a2b2_indices = np.logical_and(a_values == 'a2', b_values == 'b2')
    a2b3_indices = np.logical_and(a_values == 'a2', b_values == 'b3')

    d_values = np.empty_like(c_values)
    d_values[a1b1_indices] = np.random.normal(1, 0.75, size=a1b1_indices.sum())
    d_values[a1b2_indices] = -2 + c_values[a1b2_indices] + np.random.normal(0, 2, size=a1b2_indices.sum())
    d_values[a1b3_indices] = -1 + 3*c_values[a1b3_indices] + np.random.normal(0, 0.25, size=a1b3_indices.sum())
    d_values[a2b1_indices] = np.random.normal(2, 1, size=a2b1_indices.sum())
    d_values[a2b2_indices] = 3.5 + -1.2*c_values[a2b2_indices] + np.random.normal(0, 1, size=a2b2_indices.sum())
    d_values[a2b3_indices] = 4.8 + -2*c_values[a2b3_indices] + np.random.normal(0, 1.5, size=a2b3_indices.sum())

    return pd.DataFrame({'A': pd.Series(a_values, dtype='category'),
                         'B': pd.Series(b_values, dtype='category'),
                         'C': c_values,
                         'D': d_values
                        })

def generate_indep_hybrid_data(size, seed=0):
    np.random.seed(seed)

    d2_dict = np.asarray(['a1', 'a2'])
    d2_values = d2_dict[np.random.choice(d2_dict.size, size, p=[0.5, 0.5])]

    d3_dict = np.asarray(['b1', 'b2', 'b3'])
    d3_values = d3_dict[np.random.choice(d3_dict.size, size, p=[0.33, 0.34, 0.33])]

    d4_dict = np.asarray(['c1', 'c2', 'c3', 'c4'])
    d4_values = d4_dict[np.random.choice(d4_dict.size, size, p=[0.25, 0.25, 0.25, 0.25])]

    d5_dict = np.asarray(['d1', 'd2', 'd3', 'd4', 'd5'])
    d5_values = d5_dict[np.random.choice(d5_dict.size, size, p=[0.2, 0.2, 0.2, 0.2, 0.2])]

    d6_dict = np.asarray(['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])
    d6_values = d6_dict[np.random.choice(d6_dict.size, size, p=[0.166, 0.166, 0.166, 0.166, 0.166, 0.17])]

    c1_values = -4.2 + np.random.normal(0, 0.75, size=size)
    c2_values = np.random.normal(1, 2, size=size)
    c3_values = np.random.normal(2, 0.7, size=size)
    c4_values = np.random.normal(-3, 2.5, size=size)
    c5_values = np.random.normal(-1.2, 0.5, size=size)
    c6_values = np.random.normal(3, 1.5, size=size)

    return pd.DataFrame({'D2': pd.Series(d2_values, dtype='category'),
                         'D3': pd.Series(d3_values, dtype='category'),
                         'D4': pd.Series(d4_values, dtype='category'),
                         'D5': pd.Series(d5_values, dtype='category'),
                         'D6': pd.Series(d6_values, dtype='category'),
                         'C1': c1_values,
                         'C2': c2_values,
                         'C3': c3_values,
                         'C4': c4_values,
                         'C5': c5_values,
                         'C6': c6_values,
                        })
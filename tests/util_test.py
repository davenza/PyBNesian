import numpy as np
import pandas as pd
import pyarrow as pa


def generate_normal_data(size):
    np.random.seed(0)

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


def generate_discrete_data(size):
    np.random.seed(0)

    a_dict = np.asarray(['a1', 'a2'])
    b_dict = np.asarray(['b1', 'b2', 'b3'])
    c_dict = np.asarray(['c1', 'c2'])
    d_dict = np.asarray(['d1', 'd2', 'd3', 'd4'])

    return pd.DataFrame({'A': a_dict[np.random.randint(0, a_dict.size, size=size)],
                         'B': b_dict[np.random.randint(0, b_dict.size, size=size)],
                         'C': c_dict[np.random.randint(0, c_dict.size, size=size)],
                         'D': d_dict[np.random.randint(0, d_dict.size, size=size)]
                        }, dtype='category')
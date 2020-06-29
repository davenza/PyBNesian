import numpy as np
import pandas as pd
import pyarrow as pa
from pgm_dataset.dataset import CrossValidation


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


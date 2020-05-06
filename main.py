# import os
# os.environ['LD_LIBRARY_PATH'] = '/home/david/cpp/pgm_dataset/venv/lib/python3.6/site-packages/pyarrow'
import pyarrow as pa
# from pyarrow import libarrow
import data
# import example
import pandas as pd

if __name__ == '__main__':

    df = pd.DataFrame({'a': [0.1, 0.2, 0.3], 'b': [-23, 13, 4]})

    table = pa.Table.from_pandas(df)

    # print(example.get_array_length(table))
    print(data.is_table_py(table))
    print(data.num_rows(table))
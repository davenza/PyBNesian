# BUG: If this file and the data in pybnesian are not aligned, this does not work, I think its because of the path.append
from pathlib import Path

import numpy as np
import pandas as pd

TRUE_CLASS_LABEL = "class_label"

# Constants

SUPER_PARENT = "a"
DATA_SIZE = 10000
SAMPLE_SIZE = 100

N_NEIGHBORS = 3
SEED = 0
BN_SAVE_FOLDER_PATH = Path(__file__).parent / "models"


def generate_normal_data(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of normally distributed data with linear Gaussian relationships.
    The relationships are as follows:
    - A ~ N(3, 0.5)
    - B ~ N(2.5 + 1.65 * A, 2)
    - C ~ N(-4.2 - 1.2 * A + 3.2 * B, 0.75)
    - D ~ N(1.5 - 0.9 * A + 5.6 * B + 0.3 * C, 0.5)

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    np.random.seed(seed)

    a_array = np.random.normal(3, 0.5, size=size)
    b_array = 2.5 + 1.65 * a_array + np.random.normal(0, 2, size=size)
    c_array = (
        -4.2 - 1.2 * a_array + 3.2 * b_array + np.random.normal(0, 0.75, size=size)
    )
    d_array = (
        1.5
        - 0.9 * a_array
        + 5.6 * b_array
        + 0.3 * c_array
        + np.random.normal(0, 0.5, size=size)
    )
    df = pd.DataFrame({"a": a_array, "b": b_array, "c": c_array, "d": d_array})

    return df


def generate_normal_data_independent(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of normally distributed data with linear Gaussian relationships and independent variables.
    The relationships are as follows:
    - A ~ N(3, 0.5)
    - B ~ N(2.5, 2)
    - C ~ N(-4.2, 0.75)
    - D ~ N(1.5, 0.5)

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    np.random.seed(seed)

    a_array = np.random.normal(3, 0.5, size=size)
    b_array = np.random.normal(2.5, 2, size=size)
    c_array = np.random.normal(-4.2, 0.75, size=size)
    d_array = np.random.normal(1.5, 0.5, size=size)

    df = pd.DataFrame({"a": a_array, "b": b_array, "c": c_array, "d": d_array})
    return df


def generate_non_normal_data(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of uniformly distributed data with non-linear relationships.
    The relationships are as follows:
    - A ~ U(0, 10)
    - B ~ U(5, 15)
    - C ~ sin(A) + cos(B) + U(-1, 1)
    - D ~ exp(A / 10) + log(B + 1) + U(-0.5, 0.5)

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    np.random.seed(seed)

    # Generate uniformly distributed data
    a_values = np.random.uniform(0, 10, size)
    b_values = np.random.uniform(5, 15, size)

    # Generate non-linear relationships
    c_values = np.sin(a_values) + np.cos(b_values) + np.random.uniform(-1, 1, size)
    d_values = (
        np.exp(a_values / 10)
        + np.log(b_values + 1)
        + np.random.uniform(-0.5, 0.5, size)
    )

    # DataFrame
    df = pd.DataFrame(
        {
            "a": a_values,
            "b": b_values,
            "c": c_values,
            "d": d_values,
        }
    )
    return df


def generate_discrete_data(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of discrete data with dependent variables.
    The relationships are as follows:
    - A ~ Categorical(0.75, 0.25)
    - B ~ Categorical(0.33, 0.33, 0.34) if A = A1, else Categorical(0, 0.8, 0.2)
    - C ~ Categorical(0.5, 0.5) if A = A1 and B = B1, else Categorical(0.75, 0.25) if A = A1 and B = B2, else Categorical(0.2, 0.8) if A = A1 and B = B3, else Categorical(1, 0) if A = A2 and B = B1, else Categorical(0, 1) if A = A2 and B = B2, else Categorical(0.01, 0.99) if A = A2 and B = B3
    - D ~ Categorical(0.25, 0.25, 0.25, 0.25) if C = C1, else Categorical(0.7, 0, 0.15, 0.15) if C = C2

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    # Initialization
    np.random.seed(seed)

    a_dict = np.asarray(["A1", "A2"])
    b_dict = np.asarray(["B1", "B2", "B3"])
    c_dict = np.asarray(["C1", "C2"])
    d_dict = np.asarray(["D1", "D2", "D3", "D4"])

    a_values = a_dict[np.random.choice(a_dict.size, size, p=[0.75, 0.25])]
    b_values = np.empty(size, dtype=object)
    c_values = np.empty(size, dtype=object)
    d_values = np.empty(size, dtype=object)

    # Indices
    a1_indices = a_values == "A1"
    a2_indices = a_values == "A2"

    # Sampling B
    b_values[a1_indices] = b_dict[
        np.random.choice(b_dict.size, a1_indices.sum(), p=[0.33, 0.33, 0.34])
    ]
    b_values[a2_indices] = b_dict[
        np.random.choice(b_dict.size, a2_indices.sum(), p=[0, 0.8, 0.2])
    ]

    # Sampling C
    for i in range(size):
        if a_values[i] == "A1" and b_values[i] == "B1":
            c_values[i] = c_dict[np.random.choice(c_dict.size, p=[0.5, 0.5])]
        elif a_values[i] == "A1" and b_values[i] == "B2":
            c_values[i] = c_dict[np.random.choice(c_dict.size, p=[0.75, 0.25])]
        elif a_values[i] == "A1" and b_values[i] == "B3":
            c_values[i] = c_dict[np.random.choice(c_dict.size, p=[0.2, 0.8])]
        elif a_values[i] == "A2" and b_values[i] == "B1":
            c_values[i] = c_dict[np.random.choice(c_dict.size, p=[1, 0])]
        elif a_values[i] == "A2" and b_values[i] == "B2":
            c_values[i] = c_dict[np.random.choice(c_dict.size, p=[0, 1])]
        elif a_values[i] == "A2" and b_values[i] == "B3":
            c_values[i] = c_dict[np.random.choice(c_dict.size, p=[0.01, 0.99])]

    # Sampling D
    for i in range(size):
        if c_values[i] == "C1":
            d_values[i] = d_dict[
                np.random.choice(d_dict.size, p=[0.25, 0.25, 0.25, 0.25])
            ]
        elif c_values[i] == "C2":
            d_values[i] = d_dict[np.random.choice(d_dict.size, p=[0.7, 0, 0.15, 0.15])]

    # DataFrame
    df = pd.DataFrame(
        {
            "a": pd.Series(a_values, dtype="category"),
            "b": pd.Series(b_values, dtype="category"),
            "c": pd.Series(c_values, dtype="category"),
            "d": pd.Series(d_values, dtype="category"),
        }
    )
    return df


def generate_discrete_data_independent(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of discrete data with uniform distributions.
    The relationships are as follows:
    - A ~ Categorical(A1, A2)
    - B ~ Categorical(B1, B2, B3)
    - C ~ Categorical(C1, C2)
    - D ~ Categorical(D1, D2, D3, D4)

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    # Initialization
    np.random.seed(seed)

    a_dict = np.asarray(["A1", "A2"])
    b_dict = np.asarray(["B1", "B2", "B3"])
    c_dict = np.asarray(["C1", "C2"])
    d_dict = np.asarray(["D1", "D2", "D3", "D4"])

    # DataFrame
    df = pd.DataFrame(
        {
            "a": a_dict[np.random.randint(0, a_dict.size, size=size)],
            "b": b_dict[np.random.randint(0, b_dict.size, size=size)],
            "c": c_dict[np.random.randint(0, c_dict.size, size=size)],
            "d": d_dict[np.random.randint(0, d_dict.size, size=size)],
        },
        dtype="category",
    )
    return df


def generate_hybrid_data(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of hybrid data with discrete and continuous variables.
    The relationships are as follows:
    - A ~ Categorical(0.75, 0.25)
    - B ~ Categorical(0.3, 0.4, 0.3) if A = A1, else Categorical(0.2, 0.5, 0.3)
    - C ~ N(-4.2, 0.75)
    - D ~ N(1, 0.75) if A = A1 and B = B1, else N(-2 + C, 2) if A = A1 and B = B2, else N(-1 + 3 * C, 0.25) if A = A1 and B = B3, else N(2, 1) if A = A2 and B = B1, else N(3.5 - 1.2 * C, 1) if A = A2 and B = B2, else N(4.8 - 2 * C, 1.5) if A = A2 and B = B3

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    # Initialization
    np.random.seed(seed)

    a_dict = np.asarray(["A1", "A2"])
    a_values = a_dict[np.random.choice(a_dict.size, size, p=[0.75, 0.25])]

    b_dict = np.asarray(["B1", "B2", "B3"])
    b_values = b_dict[np.random.choice(b_dict.size, size, p=[0.3, 0.4, 0.3])]

    c_values = -4.2 + np.random.normal(0, 0.75, size=size)
    d_values = np.empty_like(c_values)

    # Indices
    a1b1_indices = np.logical_and(a_values == "A1", b_values == "B1")
    a1b2_indices = np.logical_and(a_values == "A1", b_values == "B2")
    a1b3_indices = np.logical_and(a_values == "A1", b_values == "B3")
    a2b1_indices = np.logical_and(a_values == "A2", b_values == "B1")
    a2b2_indices = np.logical_and(a_values == "A2", b_values == "B2")
    a2b3_indices = np.logical_and(a_values == "A2", b_values == "B3")

    # Sampling
    d_values[a1b1_indices] = np.random.normal(1, 0.75, size=a1b1_indices.sum())
    d_values[a1b2_indices] = (
        -2 + c_values[a1b2_indices] + np.random.normal(0, 2, size=a1b2_indices.sum())
    )
    d_values[a1b3_indices] = (
        -1
        + 3 * c_values[a1b3_indices]
        + np.random.normal(0, 0.25, size=a1b3_indices.sum())
    )
    d_values[a2b1_indices] = np.random.normal(2, 1, size=a2b1_indices.sum())
    d_values[a2b2_indices] = (
        3.5
        + -1.2 * c_values[a2b2_indices]
        + np.random.normal(0, 1, size=a2b2_indices.sum())
    )
    d_values[a2b3_indices] = (
        4.8
        + -2 * c_values[a2b3_indices]
        + np.random.normal(0, 1.5, size=a2b3_indices.sum())
    )

    # DataFrame
    df = pd.DataFrame(
        {
            "a": pd.Series(a_values, dtype="category"),
            "b": pd.Series(b_values, dtype="category"),
            "c": c_values,
            "d": d_values,
        }
    )
    return df


def generate_hybrid_data_independent(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of hybrid data with independent discrete and continuous variables.
    The relationships are as follows:
    - D2 ~ Categorical(0.5, 0.5)
    - D3 ~ Categorical(0.33, 0.34, 0.33)
    - D4 ~ Categorical(0.25, 0.25, 0.25, 0.25)
    - D5 ~ Categorical(0.2, 0.2, 0.2, 0.2, 0.2)
    - D6 ~ Categorical(0.166, 0.166, 0.166, 0.166, 0.166, 0.17)
    - C1 ~ N(-4.2, 0.75)
    - C2 ~ N(1, 2)
    - C3 ~ N(2, 0.7)
    - C4 ~ N(-3, 2.5)
    - C5 ~ N(-1.2, 0.5)
    - C6 ~ N(3, 1.5)

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    np.random.seed(seed)

    # Sampling
    d2_dict = np.asarray(["A1", "A2"])
    d2_values = d2_dict[np.random.choice(d2_dict.size, size, p=[0.5, 0.5])]

    d3_dict = np.asarray(["B1", "B2", "B3"])
    d3_values = d3_dict[np.random.choice(d3_dict.size, size, p=[0.33, 0.34, 0.33])]

    d4_dict = np.asarray(["C1", "C2", "C3", "C4"])
    d4_values = d4_dict[
        np.random.choice(d4_dict.size, size, p=[0.25, 0.25, 0.25, 0.25])
    ]

    d5_dict = np.asarray(["D1", "D2", "D3", "D4", "D5"])
    d5_values = d5_dict[
        np.random.choice(d5_dict.size, size, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    ]

    d6_dict = np.asarray(["e1", "e2", "e3", "e4", "e5", "e6"])
    d6_values = d6_dict[
        np.random.choice(
            d6_dict.size, size, p=[0.166, 0.166, 0.166, 0.166, 0.166, 0.17]
        )
    ]

    c1_values = -4.2 + np.random.normal(0, 0.75, size=size)
    c2_values = np.random.normal(1, 2, size=size)
    c3_values = np.random.normal(2, 0.7, size=size)
    c4_values = np.random.normal(-3, 2.5, size=size)
    c5_values = np.random.normal(-1.2, 0.5, size=size)
    c6_values = np.random.normal(3, 1.5, size=size)

    # DataFrame
    df = pd.DataFrame(
        {
            "D2": pd.Series(d2_values, dtype="category"),
            "D3": pd.Series(d3_values, dtype="category"),
            "D4": pd.Series(d4_values, dtype="category"),
            "D5": pd.Series(d5_values, dtype="category"),
            "D6": pd.Series(d6_values, dtype="category"),
            "C1": c1_values,
            "C2": c2_values,
            "C3": c3_values,
            "C4": c4_values,
            "C5": c5_values,
            "C6": c6_values,
        }
    )
    return df


def generate_discrete_data_classification(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of discrete data with dependent variables and a true label.
    The relationships are as follows:
    - TRUE_CLASS_LABEL ~ Categorical(0.3, 0.4, 0.3)
    - A ~ Categorical(0.6, 0.4) if TRUE_CLASS_LABEL = class1, else Categorical(0.8, 0.2) if TRUE_CLASS_LABEL = class2, else Categorical(0.5, 0.5) if TRUE_CLASS_LABEL = class3
    - B ~ Categorical(0.5, 0.3, 0.2) if TRUE_CLASS_LABEL = class1, else Categorical(0.2, 0.5, 0.3) if TRUE_CLASS_LABEL = class2, else Categorical(0.3, 0.3, 0.4) if TRUE_CLASS_LABEL = class3
    - C ~ Categorical(0.7, 0.3) if TRUE_CLASS_LABEL = class1, else Categorical(0.4, 0.6) if TRUE_CLASS_LABEL = class2, else Categorical(0.5, 0.5) if TRUE_CLASS_LABEL = class3

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    np.random.seed(seed)

    class_dict = np.asarray(["class1", "class2", "class3"])
    class_values = class_dict[
        np.random.choice(class_dict.size, size, p=[0.3, 0.4, 0.3])
    ]

    a_dict = np.asarray(["A1", "A2"])
    b_dict = np.asarray(["B1", "B2", "B3"])
    c_dict = np.asarray(["C1", "C2"])

    a_values = np.empty(size, dtype=object)
    b_values = np.empty(size, dtype=object)
    c_values = np.empty(size, dtype=object)

    # Indices
    class1_indices = class_values == "class1"
    class2_indices = class_values == "class2"
    class3_indices = class_values == "class3"

    # Sampling
    a_values[class1_indices] = a_dict[
        np.random.choice(a_dict.size, class1_indices.sum(), p=[0.6, 0.4])
    ]
    a_values[class2_indices] = a_dict[
        np.random.choice(a_dict.size, class2_indices.sum(), p=[0.8, 0.2])
    ]
    a_values[class3_indices] = a_dict[
        np.random.choice(a_dict.size, class3_indices.sum(), p=[0.5, 0.5])
    ]

    b_values[class1_indices] = b_dict[
        np.random.choice(b_dict.size, class1_indices.sum(), p=[0.5, 0.3, 0.2])
    ]
    b_values[class2_indices] = b_dict[
        np.random.choice(b_dict.size, class2_indices.sum(), p=[0.2, 0.5, 0.3])
    ]
    b_values[class3_indices] = b_dict[
        np.random.choice(b_dict.size, class3_indices.sum(), p=[0.3, 0.3, 0.4])
    ]

    c_values[class1_indices] = c_dict[
        np.random.choice(c_dict.size, class1_indices.sum(), p=[0.7, 0.3])
    ]
    c_values[class2_indices] = c_dict[
        np.random.choice(c_dict.size, class2_indices.sum(), p=[0.4, 0.6])
    ]
    c_values[class3_indices] = c_dict[
        np.random.choice(c_dict.size, class3_indices.sum(), p=[0.5, 0.5])
    ]

    # DataFrame
    df = pd.DataFrame(
        {
            TRUE_CLASS_LABEL: pd.Series(class_values, dtype="category"),
            "a": pd.Series(a_values, dtype="category"),
            "b": pd.Series(b_values, dtype="category"),
            "c": pd.Series(c_values, dtype="category"),
        }
    )
    return df


def generate_normal_data_classification(size: int, seed: int = SEED) -> pd.DataFrame:
    """Generates a DataFrame of normally distributed data with linear Gaussian relationships and a true label.
    The relationships are as follows:
    - TRUE_CLASS_LABEL ~ Categorical(0.3, 0.4, 0.3)
    - A ~ N(-4.2, 0.75)
    - B ~ N(0, 0.25) if TRUE_CLASS_LABEL = class1, else N(1, 0.5) if TRUE_CLASS_LABEL = class2, else N(2, 1) if TRUE_CLASS_LABEL = class3
    - C ~ N(-2 + 2 * B, 1) if TRUE_CLASS_LABEL = class1, else N(1 + 0.5 * B, 0.5) if TRUE_CLASS_LABEL = class2, else N(3 + 3 * B, 0.25) if TRUE_CLASS_LABEL = class3
        size (int): The sample
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    # Initialization
    np.random.seed(seed)

    class_dict = np.asarray(["class1", "class2", "class3"])
    class_values = class_dict[
        np.random.choice(class_dict.size, size, p=[0.3, 0.4, 0.3])
    ]

    a_values = -4.2 + np.random.normal(0, 0.75, size=size)

    b_values = np.empty_like(a_values)
    c_values = np.empty_like(a_values)

    # Indices
    class1_indices = class_values == "class1"
    class2_indices = class_values == "class2"
    class3_indices = class_values == "class3"

    # Sampling
    # b_values based on class_values
    b_values[class1_indices] = np.random.normal(0, 0.25, size=class1_indices.sum())
    b_values[class2_indices] = np.random.normal(1, 0.5, size=class2_indices.sum())
    b_values[class3_indices] = np.random.normal(2, 1, size=class3_indices.sum())

    # c_values based on class_values and b_values
    c_values[class1_indices] = (
        -2
        + 2 * b_values[class1_indices]
        + np.random.normal(0, 1, size=class1_indices.sum())
    )
    c_values[class2_indices] = (
        1
        + 0.5 * b_values[class2_indices]
        + np.random.normal(0, 0.5, size=class2_indices.sum())
    )
    c_values[class3_indices] = (
        3
        + 3 * b_values[class3_indices]
        + np.random.normal(0, 0.25, size=class3_indices.sum())
    )

    # DataFrame
    df = pd.DataFrame(
        {
            TRUE_CLASS_LABEL: pd.Series(class_values, dtype="category"),
            "a": a_values,
            "b": b_values,
            "c": c_values,
        }
    )
    return df


def generate_non_normal_data_classification(
    size: int, seed: int = SEED
) -> pd.DataFrame:
    """Generates a DataFrame of uniformly distributed data with non-linear relationships and a true label.
    The relationships are as follows:
    - TRUE_CLASS_LABEL ~ Categorical(0.3, 0.4, 0.3)
    - A ~ U(0, 10)
    - B ~ U(5, 15) if TRUE_CLASS_LABEL = class1, else U(10, 20) if TRUE_CLASS_LABEL = class2, else U(15, 25) if TRUE_CLASS_LABEL = class3
    - C ~ sin(A) + cos(B) + U(-1, 1) if TRUE_CLASS_LABEL = class1, else exp(A / 10) + log(B + 1) + U(-0.5, 0.5) if TRUE_CLASS_LABEL = class2, else A * B + U(-2, 2) if TRUE_CLASS_LABEL = class3

    Args:
        size (int): The sample size.
        seed (int, optional): The seed for random sampling. Defaults to 0.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    np.random.seed(seed)

    class_dict = np.asarray(["class1", "class2", "class3"])
    class_values = class_dict[
        np.random.choice(class_dict.size, size, p=[0.3, 0.4, 0.3])
    ]

    a_values = np.random.uniform(0, 10, size)

    b_values = np.empty_like(a_values)
    c_values = np.empty_like(a_values)

    # Indices
    class1_indices = class_values == "class1"
    class2_indices = class_values == "class2"
    class3_indices = class_values == "class3"

    # Sampling
    b_values[class1_indices] = np.random.uniform(5, 15, size=class1_indices.sum())
    b_values[class2_indices] = np.random.uniform(10, 20, size=class2_indices.sum())
    b_values[class3_indices] = np.random.uniform(15, 25, size=class3_indices.sum())

    c_values[class1_indices] = (
        np.sin(a_values[class1_indices])
        + np.cos(b_values[class1_indices])
        + np.random.uniform(-1, 1, size=class1_indices.sum())
    )
    c_values[class2_indices] = (
        np.exp(a_values[class2_indices] / 10)
        + np.log(b_values[class2_indices] + 1)
        + np.random.uniform(-0.5, 0.5, size=class2_indices.sum())
    )
    c_values[class3_indices] = a_values[class3_indices] * b_values[
        class3_indices
    ] + np.random.uniform(-2, 2, size=class3_indices.sum())

    # DataFrame
    df = pd.DataFrame(
        {
            TRUE_CLASS_LABEL: pd.Series(class_values, dtype="category"),
            "a": a_values,
            "b": b_values,
            "c": c_values,
        }
    )
    return df

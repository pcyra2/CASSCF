import numpy as np


def dense_to_sparse(tensor):
    np_tensor = np.array(tensor)
    non_zero_indices = list(zip(*np_tensor.nonzero()))
    return {str(index): np_tensor[index] for index in non_zero_indices}

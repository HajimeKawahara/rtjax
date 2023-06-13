import numpy as np
from rtjax.naivebmat import create_block_matrix 

def test_create_block_matrix():
    A = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    B = [np.array([[9, 10], [11, 12]]), np.array([[13, 14], [15, 16]])]
    
    expected_output = np.array([[1, 2, 9, 10, 0, 0], 
                                [3, 4, 11, 12, 0, 0], 
                                [0, 0, 5, 6, 13, 14], 
                                [0, 0, 7, 8, 15, 16]])

    np.testing.assert_array_equal(create_block_matrix(A, B), expected_output)



if __name__ == "__main__":
    test_create_block_matrix()


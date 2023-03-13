import numpy as np

# cam1 = np.array([[1, 2, 3, 4],
#                  [2, 3, 1, 6],
#                  [4, 3, 1, 1],
#                  [3, 4, 5, 6]]) 
                      
# cam2 = np.array([[3, 1, 5 ,6],
#                  [6, 2, 3, 1],
#                  [3, 2, 1, 1],
#                  [2, 1, 6, 7]])

# cam3 = cam2.T * 100
# table = [cam1, cam2, cam3]

def iterative_elimination(table):

    list_mean_max_row_cams = []
    
    for cam in table:
        max_row = np.max(cam, axis=1)
        mean_max_row = np.mean(max_row)
        list_mean_max_row_cams.append(mean_max_row)
    
    index_cam = np.argmax(list_mean_max_row_cams)
    
    new_table = table[index_cam].astype(float)
    
    mapping = np.zeros(table.shape(0), dtype=int)
    
    for i in range(table.shape(0)):
    
        max_index = np.unravel_index(new_table.argmax(), new_table.shape)
        row, column = max_index
        mapping[row] = column
        new_table[row, :] = -np.inf
        new_table[:, column] = -np.inf 
    
    return mapping
    
    
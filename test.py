import numpy as np

# all_pos = np.random.uniform(-10, +10, (10, 2))
# print(all_pos)
# distance = np.linalg.norm(all_pos - all_pos[2],axis=1)
# print(distance)
#
#
# neighbors = list(np.where(distance<1)[0])
# print(neighbors)
# neighbors.remove(2)
# print(neighbors)
# pro_neighbors = list(np.where((distance<1.5) & (distance>1))[0])
# print(pro_neighbors)
a = [[]] * 6

a[0] = [1,2,3]

b = [value for value in a if len(value)!=0]
print(b)

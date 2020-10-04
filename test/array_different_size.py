import numpy as np

a = np.array([[1,1], [1,1]])
a_shape = a.shape
print('a is', a)

a.resize((4,5))
#high_dima = np.reshape(a, (2, 3))
print('Resized a is', a)

a.resize(a_shape)
print('Restored a is', a)

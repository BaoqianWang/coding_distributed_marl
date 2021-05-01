import shapely
from shapely.geometry.point import Point
from shapely.geometry import Polygon

p_cir = Point(0, 0)
cir = p_cir.buffer(1.414)
point_circle = cir.exterior.coords
p_cir_poly = Polygon(point_circle)

p1=Polygon([(0,0),(1,1),(1,0)])
p2=Polygon([(0,1),(1,0),(1,1)])
p3=p1.intersection(p2)
print(p3) # result: POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0.5))
print(p3.area) # result: 0.25

p4 = p_cir_poly.intersection(p1)
print(p4.area)

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
# a = [[0]] * 6
#
# for i in range(3):
#     a[i] = [i]
#     print(a)

# b = [i for i in range(8) if i >4]
# print(b)
#
# print(np.random.uniform(0,1))

#b = [1,1,1, -2, -3]
#.remove(1)
# b = 2 if np.random.uniform(0,1)> 0.5 4 else
# c =
# print(b)

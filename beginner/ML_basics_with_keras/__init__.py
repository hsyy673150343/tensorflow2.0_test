import numpy as np
from collections import deque

arrary = np.array([[3,4,5],[6,7,8],[9,11,12]])
print(arrary)
result = []
count = 0
for i in arrary:
    count += 1
    if count == 1:
        a = deque(arrary[0])
        a.appendleft(arrary[0][0])
        a.append(arrary[0][2])
        result.append(list(a))
        result.append(list(a))
    elif count == 2:
        b = deque(arrary[1])
        b.appendleft(arrary[1][0])
        b.append(arrary[1][2])
        result.append(list(b))
    elif count == 3:
        c = deque(arrary[2])
        c.appendleft(arrary[2][0])
        c.append(arrary[2][2])
        result.append(list(c))
        result.append(list(c))

print(np.array(result))
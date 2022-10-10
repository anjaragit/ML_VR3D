import sys
import ezdxf
from collections import Counter
from matplotlib import pyplot as plt

doc = ezdxf.readfile("data/DX_J35_ASS.dxf")
msp = doc.modelspace()
lwschema = msp.query('LWPOLYLINE')
print('nb chambre :' ,len(lwschema))

mylist = []
counted = []
sommet = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10]

for line in lwschema:
    print('\n\nID = ', line.dxf.layer)
    print('nb sommet = ',len(line))
    mylist.append(len(line))
counting = Counter(mylist)
for s in sommet :
    c = counting[s]
    counted.append(c)
print(counted)
# plt.plot(sommet, counted)
plt.bar(sommet, counted, color ='maroon',
        width = 0.4)
plt.show()
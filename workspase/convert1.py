target = 49
amari = []

while target != 0:
    amari.append(target % 2)
    target = target // 2

amari.reverse()
print(amari)

"""
for i in range(a // 2 > 0):
    a = a // 2
    b = a % 2
    print(str(b))
"""
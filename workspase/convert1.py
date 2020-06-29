target = 49
amari1 = []
amari2 = []
amari3 = []

while target != 0:
    amari1.append(target % 2)
    target = target // 2

amari1.reverse()
print(amari1)

while target != 0:
    amari2.append(target % 8)
    target = target // 8
amari2.reverse()
print(amari2)

while target != 0:
    amari3.append(target % 16)
    target = target // 16
amari3.reverse()
print(amari3)
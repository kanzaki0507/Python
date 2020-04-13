import sys

seireki = input('西暦を入力してください:')
if not seireki:
    print('整数を入力して下さい')
    sys.exit()

if int(seireki) < 1868:
    sys.exit()

elif int(seireki) < 1912:
    print("明治" + str(int(seireki)-1867) + "年")

elif int(seireki) < 1926:
    print("大正" + str(int(seireki)-1911) + "年")

elif int(seireki) < 1989:
    print("昭和" + str(int(seireki)-1925) + "年")

elif int(seireki) < 2019:
    print("平成" + str(int(seireki)-1988) + "年")

else:
    print("令和" + str(int(seireki)-2018) + "年")
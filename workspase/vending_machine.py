import sys

insert_price = input('insert:')
if not insert_price.isdecimal():
    print('整数を入力して下さい')
    sys.exit()

product_price = input('product:')
if not product_price.isdecimal():
    print('整数を入力して下さい')
    sys.exit()

change = int(insert_price) - int(product_price)
print("お釣り:" + str(change))

if change < 0:
    print('金額が不足しています')
    sys.exit()

coin = [5000, 1000, 500, 100, 50, 10, 5, 1]

for i in coin:
    r = change // i
    change %= i
    print(str(i) + ':' + str(r))
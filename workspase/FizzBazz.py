for i in range(1, 50):
    if ((i % 3 == 0) & (i % 5 == 0)):
        print("FizzBazz", end=",")
    elif (i % 3) == 0:
        print("Fizz", end=",")
    elif (i % 5) == 0:
        print("Bazz", end=",")
    else:
        print(i, end=",")
print("")

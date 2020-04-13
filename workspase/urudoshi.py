start = 1950
end = 2050

for i in range(start, end+1):
    if (i % 4 == 0):
        print(str(i)+("(うるう年)"), end=' ')
    elif (i % 100 == 0) and (i % 400 != 0):
        print(i, end=' ')
    else :
        print(i, end=' ')
print("")
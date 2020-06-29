def search(data, target):
    for i in range(len(data)):
        if data[i] == target:
            return i

def search_nibun(data, target):
    start, end = 0, len(data)

    while start <= end:
        mid = (start + end) // 2

        if data[mid] == target:
            print("要素{}にデータ{}を探索しました。".format(mid, target))
            return mid
        elif data[mid] < target:
            start = mid + 1
        else:
            end = mid - 1
        # return -1

def bubble_sort(data):
    for i in range(len(data)-1, 0, -1):
        for j in range(i):
            # print(i, j, len(data)-1)
            if data[j] > data[j+1]:
                print(data[j], data[j+1])
                data[j], data[j+1] = data[j+1], data[j]

def insert_sort(data):
    for i in range(0, len(data)):
        for j in range(i-1, -1, -1):
            if data[j] > data[j+1]:
                print(i, data[j], data[j+1])
                data[j], data[j+1] = data[j+1], data[j]
            else:
                break

def select_sort(data):
    for i in range(0, len(data)-1):
        min = i
        for j in range(i+1, len(data)):
            if data[min] > data[j]:
                min = j
            # print("min:{} i:{} j:{}".format(data[min], data[i], data[j]))
        data[min], data[i] = data[i], data[min]
        print(data)

def shell_sort(data):
    gaps = [5, 3, 1]
    for gap in gaps:
        for i in range(0, len(data), gap):
            for j in range(i-gap, -1, -gap): #"-gap"を"-i"にしたらうまく動かない
                print("i:{} j:{} j+gaps:{} data[j]={} data[j+gap]={}".format(i, j, j+gap, data[j], data[j+gap]))
                if data[j] > data[j+gap]:
                    data[j], data[j+gap] = data[j+gap], data[j]
                    # print(data)
                else:
                    break
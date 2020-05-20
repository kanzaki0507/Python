def min_heapify(array, i):
    left = 2 * i + 1
    right = 2 * i + 2
    length = len(array) - 1
    smallest = i

    if left <= length and array[i] > array[left]:
        smallest = left
    if right <= length and array[smallest] > array[right]:
        smallest = right
    if smallest != i:
        array[i], array[smallest] = array[smallest], array[i]
        min_heapify(array, smallest)
    print("data : " + str(data))
    #print("smallest : " + str(smallest))

def max_heap(data, k):

    left = 2 * k + 1
    right = 2 * k + 2
    length = len(data)-1
    largest = k

    if left <= length and data[left] > data[k]:
        largest = left
    if right <= length and data[right] >  data[largest]:
        largest = right
    if largest != k:
        data[k], data[largest] = data[largest], data[k]
        max_heap(data, largest)

    print(data)

def sort(data):

    data = data.copy()
    ## Build Heap Tree
    for i in reversed(range(len(data)//2)):
       min_heapify(data, i)

    sort_data = []
    
    for _ in range(len(data)):
        data[0], data[-1] = data[-1], data[0]
        sort_data.append(data.pop())
        min_heapify(data, 0)
        # print(data)

    return print("Heap Sort : " + str(sort_data))

data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]
sort(data)
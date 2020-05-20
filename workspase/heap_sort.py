"""
data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

for i in range(len(data)):
    j = i

    while (j > 0) and (data[(j-1)//2] < data[j]):
        data[(j-1)//2], data[j] = data[j], data[(j-1)//2]
        j = (j - 1) // 2
    print(data)
# print(len(data))
"""

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

        

data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]
# k = (len(data)-1) // 2

## Build Heap tree
for i in reversed(range(len(data)//2)):
   min_heapify(data, i)

## Heap Sort

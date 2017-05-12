

def quicksort(data):
    data_len = len(data)
    if data == []:
        return data

    if data_len == 1:
        return data

    pivot_idx = int(data_len / 2)
    pivot = data[pivot_idx]

    left = []
    right = []
    cnt = 0
    for i in data:

        if i <= pivot and cnt != pivot_idx:
            left.append(i)

        else:
            if cnt != pivot_idx:
                right.append(i)

        cnt+=1

    return quicksort(left) + [pivot] + quicksort(right)



test1 = [2,6,3,2,4,5,6,4,2,2,4,5,6,3]
test2 = [10,22,63,15, 3]

print(quicksort(test1))
print(quicksort(test2))
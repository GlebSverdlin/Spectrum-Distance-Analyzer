import random

a = [[1, 2, 3], [3, 4, 6]]


def matrix(m, n):
    matrix = []
    for i in range(0, m):
        row = []
        for j in range(0, n):
            row.append(random.randint(0, 9))
        matrix.append(row)
    return matrix


def matrix_print(matrix):
    k = len(matrix)
    i = 0

    while i < k:
        print(f"{matrix[i]}")
        i += 1


def matrix_shape(matrix):
    m = len(matrix)
    n = len(matrix[0])
    return m, n


def matrix_transpose(matrix):
    matrix_t = []
    m, n, square = matrix_shape(matrix)
    print(square)
    for j in range(0, n):
        row = []
        for i in range(0, m):
            row.append(matrix[i][j])
        matrix_t.append(row)
    return matrix_t


def matrix_compare(matrix_1, matrix_2):
    m_1, n_1 = matrix_shape(matrix_1)
    m_2, n_2 = matrix_shape(matrix_2)
    if m_1 != m_2 or n_1 != n_2:
        print("Error: incompatible shapes.")
        return False
    for i in range(0, n_1):
        for j in range(0, m_1):
            if matrix_1[j][i] != matrix_2[j][i]:
                return False
    return True


def matrix_square(matrix):
    m, n = matrix_shape(matrix)
    if m > n:
        k = 0
        while k < (m - n):
            for j in range(0, m):
                matrix[j].append(None)
            k += 1

    if n > m:
        k = 0
        while k < (n - m):
            row = []
            for j in range(0, m+1):
                row.append(None)
            matrix.append(row)
            k += 1
    return matrix


def matrix_clear(matrix):
    m, n = matrix_shape(matrix)
    delete_arr = []
    j = 0
    i = 0
    while j < m:
        while i < n:
            if matrix[j][i] is None:
                delete_arr.append(j)
                delete_arr.append(i)
            i += 1
        i = 0
        j += 1
    print("to delete:", delete_arr)
    for d in range(0, len(delete_arr), 2):
        print(d, delete_arr[d], delete_arr[d+1])
        j = delete_arr[d]
        i = delete_arr[d+1]
        print(j, i)
        print(matrix[j][i])
    return matrix


def transpose(matrix):
    m, n = matrix_shape(matrix)
    print(m, n)
    if m != n:
        matrix_square(matrix)
        matrix_print(matrix)
    n_min = 0
    for j in range(0, m):
        for i in range(n_min, n):
            element = matrix[j][i]
            matrix[j][i] = matrix[i][j]
            matrix[i][j] = element
        n_min += 1

    return matrix


b = matrix(1, 2)

matrix_print(b)
print('---b square:---')
matrix_print(matrix_square(b))
print('---b cleared:---')
matrix_print(matrix_clear(b))

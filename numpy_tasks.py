"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    n = matrices[0].shape[0]
    result = np.zeros((n, 1))
    for i in range(len(matrices)):
        result += matrices[i] @ vectors[i]
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > threshold:
                result[i, j] = 1
    return result


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for i in range(matrix.shape[0]):
        row = matrix[i]
        unique = []
        for v in row:
            if v not in unique:
                unique.append(v)
        result.append(unique)
    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for j in range(matrix.shape[1]):
        unique = []
        for i in range(matrix.shape[0]):
            v = matrix[i, j]
            if v not in unique:
                unique.append(v)
        result.append(unique)
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    result = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                result[i, j] = first
            else:
                result[i, j] = second
    return result


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.zeros((image_height, image_width, 3))
    for i in range(image_height):
        for j in range(image_width):
            image[i, j] = background_color

    top = (image_height - height) // 2
    left = (image_width - width) // 2
    for i in range(top, top + height):
        for j in range(left, left + width):
            image[i, j] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.zeros((image_height, image_width, 3))
    for i in range(image_height):
        for j in range(image_width):
            image[i, j] = background_color

    x0 = image_width / 2
    y0 = image_height / 2
    for i in range(image_height):
        for j in range(image_width):
            if ((j - x0) ** 2) / (semi_axis_x ** 2) + ((i - y0) ** 2) / (semi_axis_y ** 2) <= 1:
                image[i, j] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    n = len(values)

    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))

    maxima = []
    minima = []
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append(i)

    smoothed = []
    for i in range(n - window + 1):
        s = 0
        for j in range(i, i + window):
            s += values[j]
        smoothed.append(s / window)

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=np.array(smoothed),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1
    result = np.zeros((len(labels), class_count))
    for i in range(len(labels)):
        result[i, labels[i]] = 1
    return result

def test_sum_prod_simple():
    mats = np.array([np.eye(2), np.eye(2) * 2])
    vecs = np.array([[[1], [2]], [[3], [4]]])
    result = sum_prod(MatrixVectorBatchInput(mats, vecs))
    # I*[1,2] + 2I*[3,4] = [1,2] + [6,8] = [7,10]
    assert np.allclose(result, [[7], [10]])
    assert result.shape == (2, 1)


def test_sum_prod_one_matrix():
    mats = np.array([np.array([[2.0, 0.0], [0.0, 3.0]])])
    vecs = np.array([[[1.0], [1.0]]])
    result = sum_prod(MatrixVectorBatchInput(mats, vecs))
    assert np.allclose(result, [[2], [3]])


# ---------- Задача 2. binarize ----------

def test_binarize_basic():
    m = np.array([[0.1, 0.5], [0.9, 0.2]])
    result = binarize(BinarizeInput(m, 0.4))
    assert np.array_equal(result, [[0, 1], [1, 0]])


def test_binarize_strict_greater():
    # ровно threshold — не больше, значит 0
    m = np.array([[0.5, 0.6]])
    result = binarize(BinarizeInput(m, 0.5))
    assert np.array_equal(result, [[0, 1]])


def test_binarize_does_not_change_input():
    m = np.array([[0.1, 0.9]])
    binarize(BinarizeInput(m, 0.5))
    assert np.array_equal(m, [[0.1, 0.9]])


# ---------- Задача 3. unique_rows / unique_columns ----------

def test_unique_rows():
    m = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_rows(MatrixInput(m)) == [[1, 2], [3]]


def test_unique_rows_order_preserved():
    m = np.array([[3, 1, 3, 2, 1]])
    assert unique_rows(MatrixInput(m)) == [[3, 1, 2]]


def test_unique_columns():
    m = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_columns(MatrixInput(m)) == [[1, 3], [1, 3], [2, 3]]


# ---------- Задача 4. matrix_statistics ----------

def test_matrix_statistics_shapes():
    data = RandomMatrixInput(rows=4, columns=5, mean=0.0, std=1.0, seed=42)
    stats = matrix_statistics(data)
    assert stats.matrix.shape == (4, 5)
    assert stats.row_means.shape == (4,)
    assert stats.column_means.shape == (5,)
    assert stats.row_variances.shape == (4,)
    assert stats.column_variances.shape == (5,)


def test_matrix_statistics_reproducible():
    data = RandomMatrixInput(rows=3, columns=3, mean=5.0, std=2.0, seed=1)
    s1 = matrix_statistics(data)
    s2 = matrix_statistics(data)
    assert np.allclose(s1.matrix, s2.matrix)


def test_matrix_statistics_values():
    data = RandomMatrixInput(rows=3, columns=3, mean=0.0, std=1.0, seed=0)
    stats = matrix_statistics(data)
    assert np.allclose(stats.row_means, stats.matrix.mean(axis=1))
    assert np.allclose(stats.column_means, stats.matrix.mean(axis=0))
    assert np.allclose(stats.row_variances, stats.matrix.var(axis=1))
    assert np.allclose(stats.column_variances, stats.matrix.var(axis=0))


# ---------- Задача 5. chess ----------

def test_chess_example():
    result = chess(ChessInput(3, 4, 0, 1))
    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    assert np.array_equal(result, expected)


def test_chess_top_left_is_first():
    result = chess(ChessInput(2, 2, 7, 9))
    assert result[0, 0] == 7
    assert result[0, 1] == 9
    assert result[1, 0] == 9
    assert result[1, 1] == 7


def test_chess_one_cell():
    result = chess(ChessInput(1, 1, 5, 8))
    assert np.array_equal(result, [[5]])


# ---------- Задача 6. draw_rectangle / draw_ellipse ----------

def test_rectangle_shape_and_colors():
    data = RectangleInput(
        width=2, height=2,
        image_height=6, image_width=6,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )
    img = draw_rectangle(data)
    assert img.shape == (6, 6, 3)
    # центр — цвет фигуры
    assert np.array_equal(img[3, 3], [255, 0, 0])
    # угол — фон
    assert np.array_equal(img[0, 0], [0, 0, 0])


def test_ellipse_shape_and_center():
    data = EllipseInput(
        semi_axis_x=3, semi_axis_y=2,
        image_height=10, image_width=10,
        shape_color=(0, 255, 0),
        background_color=(255, 255, 255),
    )
    img = draw_ellipse(data)
    assert img.shape == (10, 10, 3)
    # центр точно внутри
    assert np.array_equal(img[5, 5], [0, 255, 0])
    # угол снаружи
    assert np.array_equal(img[0, 0], [255, 255, 255])


# ---------- Задача 7. analyze_time_series ----------

def test_time_series_basic():
    values = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    ts = analyze_time_series(TimeSeriesInput(values, 2))
    assert np.isclose(ts.mean, 3.0)
    assert np.isclose(ts.variance, values.var())
    assert np.isclose(ts.std, values.std())
    assert ts.local_maxima_indices == [1, 3]
    assert ts.local_minima_indices == [2]


def test_time_series_moving_average():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    ts = analyze_time_series(TimeSeriesInput(values, 2))
    assert np.allclose(ts.moving_average, [1.5, 2.5, 3.5])


def test_time_series_no_extrema():
    values = np.array([1.0, 2.0, 3.0])
    ts = analyze_time_series(TimeSeriesInput(values, 1))
    assert ts.local_maxima_indices == []
    assert ts.local_minima_indices == []


def test_time_series_window_equals_length():
    values = np.array([1.0, 2.0, 3.0])
    ts = analyze_time_series(TimeSeriesInput(values, 3))
    assert np.allclose(ts.moving_average, [2.0])


# ---------- Задача 8. one_hot ----------

def test_one_hot_class_count_none():
    labels = np.array([0, 2, 3, 0])
    result = one_hot(OneHotInput(labels, None))
    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    assert np.array_equal(result, expected)


def test_one_hot_explicit_class_count():
    labels = np.array([0, 1])
    result = one_hot(OneHotInput(labels, 5))
    assert result.shape == (2, 5)
    assert np.array_equal(result[0], [1, 0, 0, 0, 0])
    assert np.array_equal(result[1], [0, 1, 0, 0, 0])


def test_one_hot_single_label():
    labels = np.array([0])
    result = one_hot(OneHotInput(labels, None))
    assert np.array_equal(result, [[1]])
"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    count=0
    for i in text:
        if i in ("aeiou"):
            count+=1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for i in str(number):
            product *= int(i)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    result=0
    for i in range (len(predicted)):
        result +=(predicted[i]-expected[i])**2
    return result / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = ""
    d = 2
    while d * d <= number:
        p = 0
        while number % d == 0:
            number = number // d
            p += 1
        if p == 1:
            result += "(" + str(d) + ")"
        elif p > 1:
            result += "(" + str(d) + "**" + str(p) + ")"
        d += 1
    if number > 1:
        result += "(" + str(number) + ")"
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    text = str(data.value)
    n = len(text)
    g = (n % 2 == 0)
    half = n // 2
    left = 0
    right = 0
    for i in range(half - g):
        left += int(text[i])
    for i in range(half + 1, n):
        right += int(text[i])
    return left == right


# count_vowels
assert count_vowels(TextInput("Hello")) == 2
assert count_vowels(TextInput("AEIOU")) == 5
assert count_vowels(TextInput("xyz")) == 0

# has_unique_characters
assert has_unique_characters(TextInput("abc")) is True
assert has_unique_characters(TextInput("abca")) is False
assert has_unique_characters(TextInput("")) is True

# count_one_bits
assert count_one_bits(PositiveIntegerInput(0)) == 0
assert count_one_bits(PositiveIntegerInput(7)) == 3
assert count_one_bits(PositiveIntegerInput(8)) == 1

# multiplicative_persistence
assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
assert multiplicative_persistence(PositiveIntegerInput(999)) == 4

# mse
assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0
assert mse(VectorPairInput([1, 2], [3, 4])) == 4.0

# prime_factorization
assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"

# pyramid
assert pyramid(PositiveIntegerInput(1)) == 1
assert pyramid(PositiveIntegerInput(5)) == 2
assert pyramid(PositiveIntegerInput(4)) == "It is impossible"

# is_balanced_number
assert is_balanced_number(PositiveIntegerInput(1234006)) is True
assert is_balanced_number(PositiveIntegerInput(123456)) is False
assert is_balanced_number(PositiveIntegerInput(119011)) is True
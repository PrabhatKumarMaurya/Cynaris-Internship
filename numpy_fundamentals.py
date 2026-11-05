
import numpy as np


# Normalize function with type hints
def normalize(data: list[float]) -> list[float]:
    """
    Return min-max scaled values in the range 0 to 1.

    If all values are identical, return zeros
    to prevent division by zero.
    """
    if not data:
        return []

    lo = min(data)
    hi = max(data)

    if hi == lo:
        return [0.0 for _ in data]

    return [(x - lo) / (hi - lo) for x in data]


# Test cases for normalize function
def test_normalize():
    assert normalize([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert normalize([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]
    assert normalize([7, 7, 7]) == [0.0, 0.0, 0.0]
    assert normalize([42]) == [0.0]
    assert normalize([10, 15.5, 20]) == [0.0, 0.55, 1.0]

    print("All normalize tests passed!")


# Main program
def main():
    # 1. 1D Array
    a = np.array([10, 20, 30, 40])

    # 2. 2D Array
    b = np.array([[1, 2], [3, 4]])

    # 3. 3D Array
    c = np.array([[[1, 2], [3, 4]]])

    print("1D Shape:", a.shape)
    print("2D Shape:", b.shape)
    print("3D Shape:", c.shape)

    # 4. Vectorized Operation
    print("Addition:", a + 10)

    # 5. Broadcasting
    print("Broadcasting:", a * 2)

    # 6. Matrix Multiplication
    print("Matrix Multiplication:\n", b @ b)

    # 7. Statistics
    print("Mean:", np.mean(a))
    print("Standard Deviation:", np.std(a))
    print("Correlation:", np.corrcoef(a, [20, 30, 40, 50]))


if __name__ == "__main__":
    test_normalize()
    main()

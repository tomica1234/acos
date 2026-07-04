from feature import add

def test_add_positive_integers():
    assert add(2, 3) == 5

def test_add_negative_integers():
    assert add(-1, 1) == 0
    assert add(-5, -3) == -8

def test_add_zero():
    assert add(0, 0) == 0
    assert add(5, 0) == 5

def test_add_floats():
    assert add(1.5, 2.5) == 4.0

def test_add_large_numbers():
    assert add(10**18, 10**18) == 2 * 10**18
"""Test the autograd engine."""

from strands_microgpt.engine import Value


def test_value_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0


def test_value_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0


def test_value_backward_simple():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a
    c.backward()
    # dc/da = b + 1 = 4.0
    assert a.grad == 4.0
    # dc/db = a = 2.0
    assert b.grad == 2.0


def test_value_relu():
    a = Value(-3.0)
    b = Value(3.0)
    assert a.relu().data == 0.0
    assert b.relu().data == 3.0


def test_value_exp_log():
    import math

    a = Value(1.0)
    b = a.exp()
    assert abs(b.data - math.e) < 1e-5

    c = b.log()
    assert abs(c.data - 1.0) < 1e-5


def test_value_pow():
    a = Value(3.0)
    b = a**2
    assert b.data == 9.0
    b.backward()
    assert a.grad == 6.0  # d(x^2)/dx = 2x = 6


def test_value_division():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    assert c.data == 2.0


def test_value_neg():
    a = Value(5.0)
    b = -a
    assert b.data == -5.0


def test_chain_rule():
    """Test multi-step chain rule."""
    x = Value(2.0)
    y = x * x * x  # x^3
    y.backward()
    # dy/dx = 3x^2 = 12.0
    assert abs(x.grad - 12.0) < 1e-5

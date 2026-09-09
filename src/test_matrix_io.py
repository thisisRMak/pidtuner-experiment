"""Unit tests for matrix_io.py's parse_matlab_literal — MATLAB-style matrix
literal parsing (e.g. "[1 0; 0 1]") plus Python/numpy-style nested-list
syntax (e.g. "[[1,0],[0,1]]") — and format_matlab_literal, its reverse.

Run with:
    python test_matrix_io.py
or:
    python -m unittest test_matrix_io -v
"""

import unittest

import numpy as np

from matrix_io import format_matlab_literal, parse_matlab_literal


class TestParseMatlabLiteralMatlabStyle(unittest.TestCase):
    def test_bracketed_semicolon_rows(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[1 0; 0 1]"), [[1.0, 0.0], [0.0, 1.0]])

    def test_bracketed_comma_and_space_mixed(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[1,0; 0,1]"), [[1.0, 0.0], [0.0, 1.0]])

    def test_no_outer_brackets(self):
        np.testing.assert_allclose(
            parse_matlab_literal("1, 0; 0, 1"), [[1.0, 0.0], [0.0, 1.0]])

    def test_single_row(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[1 2 3]"), [[1.0, 2.0, 3.0]])


class TestParseMatlabLiteralPythonListStyle(unittest.TestCase):
    def test_nested_list(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[[1,0],[0,1]]"), [[1.0, 0.0], [0.0, 1.0]])

    def test_nested_single_row(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[[1, 2, 3]]"), [[1.0, 2.0, 3.0]])

    def test_flat_list_becomes_single_row(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[1, 2, 3]"), [[1.0, 2.0, 3.0]])

    def test_matches_equivalent_matlab_style_input(self):
        np.testing.assert_allclose(
            parse_matlab_literal("[[1,0],[0,1]]"),
            parse_matlab_literal("[1 0; 0 1]"))


class TestParseMatlabLiteralMalformed(unittest.TestCase):
    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError):
            parse_matlab_literal("")

    def test_matlab_style_ragged_rows_rejected(self):
        with self.assertRaises(ValueError):
            parse_matlab_literal("[1 2; 3]")

    def test_matlab_style_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            parse_matlab_literal("not a matrix")

    def test_python_list_style_ragged_rows_rejected(self):
        with self.assertRaises(ValueError):
            parse_matlab_literal("[[1,2],[3]]")

    def test_python_list_style_ragged_rows_rejected_other_direction(self):
        with self.assertRaises(ValueError):
            parse_matlab_literal("[[1,2],[3,4,5]]")


class TestFormatMatlabLiteral(unittest.TestCase):
    def test_identity_2x2(self):
        self.assertEqual(format_matlab_literal(np.eye(2)), "[1 0; 0 1]")

    def test_single_row(self):
        self.assertEqual(
            format_matlab_literal(np.array([[1.0, 2.0, 3.0]])), "[1 2 3]")

    def test_1d_array_treated_as_single_row(self):
        self.assertEqual(
            format_matlab_literal(np.array([1.0, 2.0, 3.0])), "[1 2 3]")

    def test_negative_and_multidigit_entries(self):
        self.assertEqual(
            format_matlab_literal(np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]])),
            "[0 1 0; 0 0 1; -6 -11 -6]")

    def test_non_integer_entries(self):
        self.assertEqual(
            format_matlab_literal(np.array([[1.5, -0.25]])), "[1.5 -0.25]")

    def test_empty_array_rejected(self):
        with self.assertRaises(ValueError):
            format_matlab_literal(np.zeros((0, 0)))


class TestFormatMatlabLiteralRoundTrip(unittest.TestCase):
    def _check(self, arr):
        np.testing.assert_allclose(
            parse_matlab_literal(format_matlab_literal(arr)), arr)

    def test_identity_3x3(self):
        self._check(np.eye(3))

    def test_rectangular(self):
        self._check(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_single_row(self):
        self._check(np.array([[1.0, 2.0, 3.0]]))

    def test_1d_array(self):
        # parse_matlab_literal always returns 2-D, reshaping a 1-D input
        # into a single row — so the round trip is checked against that
        # reshaped form, not the original 1-D array.
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            parse_matlab_literal(format_matlab_literal(arr)),
            arr.reshape(1, -1))

    def test_negative_entries(self):
        self._check(np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]]))

    def test_non_integer_entries(self):
        self._check(np.array([[1.5, -0.25, 1.0 / 3.0], [2.71828, 0.0, -100.125]]))

    def test_single_entry(self):
        self._check(np.array([[42.0]]))


if __name__ == "__main__":
    unittest.main()

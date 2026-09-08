"""Unit tests for matrix_io.py's parse_matlab_literal — MATLAB-style matrix
literal parsing (e.g. "[1 0; 0 1]") plus Python/numpy-style nested-list
syntax (e.g. "[[1,0],[0,1]]").

Run with:
    python test_matrix_io.py
or:
    python -m unittest test_matrix_io -v
"""

import unittest

import numpy as np

from matrix_io import parse_matlab_literal


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


if __name__ == "__main__":
    unittest.main()

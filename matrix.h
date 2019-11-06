#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <cstdio>
#include <cassert>
#include <vector>
using std::vector;

typedef vector< vector<double> > Matrix;

inline double dot(const Matrix& A, const Matrix& B) {
	assert(A.size() == B.size() && A[0].size() == B[0].size());

	double ret = 0.0;
	int r = A.size(), c = A[0].size();

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			ret += A[i][j] * B[i][j];
		}
	}

	return ret;
}

inline Matrix operator*(const Matrix& A, double x) {
	int r = A.size(), c = A[0].size();
	Matrix ret(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			ret[i][j] = A[i][j] * x;
		}
	}

	return ret;
}

inline double sum(const Matrix& A) {
	double ret = 0.0;
	int r = A.size(), c = A[0].size();

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			ret += A[i][j];
		}
	}

	return ret;
}

inline Matrix operator+(const Matrix& A, const Matrix& B) {
	int r = A.size(), c = A[0].size();
	Matrix ret(r, vector<double>(c, 0.0));

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			ret[i][j] = A[i][j] + B[i][j];
		}
	}

	return ret;
}

inline void blockAdd(Matrix& A, int leftmost, int topmost, const Matrix& B) {
	int r = B.size(), c = B[0].size();

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			A[leftmost + i][topmost + j] += B[i][j];
		}
	}
}

inline Matrix block(const Matrix& A, int leftmost, int topmost, int size) {
	Matrix ret(size, vector<double>(size, 0.0));

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			ret[i][j] = A[leftmost + i][topmost + j];
		}
	}

	return ret;
}

inline Matrix maxPadding(int pad, const Matrix& A, int size, const Matrix& thisY, const Matrix& prevX) {
	Matrix ret(pad, vector<double>(pad, 0.0));

	for (int i = 0; i < pad; ++i) {
		for (int j = 0; j < pad; ++j) {
			if (prevX[i][j] == thisY[i / size][j / size]) {
				ret[i][j] = A[i / size][j / size];
			}
		}
	}

	return ret;
}

inline void print(const Matrix& A) {
	int r = A.size(), c = A[0].size();

	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			fprintf(stderr, "%.2f%c", A[i][j], j == c - 1 ? '\n' : ' ');
		}
	}
}

#endif

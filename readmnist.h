#pragma once

#ifndef READMNIST_H
#define READMNIST_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix.h"
#define __TRAIN_IMG__ ("train-images.idx3-ubyte")
#define __TRAIN_LAB__ ("train-labels.idx1-ubyte")
#define __TEST_IMG__ ("t10k-images.idx3-ubyte")
#define __TEST_LAB__ ("t10k-labels.idx1-ubyte")

typedef unsigned char uchar;

struct mnistNode {
	Matrix img;
	Matrix label;
	mnistNode() { 
		img = Matrix(32, vector<double>(32, 0.0));
		label = Matrix(1, vector<double>(10, 0.0));
	}
};

struct mnist {
	int trainCases, testCases;
	vector<mnistNode> trainSet, testSet;
};

constexpr int decode(int x) {
	uchar ch[4] = { x & 255, (x >> 8) & 255, (x >> 16) & 255, (x >> 24) & 255 };
	return ((int)ch[0] << 24) + ((int)ch[1] << 16) + ((int)ch[2] << 8) + (int)ch[3];
}

inline int readImage(const char *filename, int& testcases, vector<mnistNode>& nSet) {
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		fprintf(stderr, "Open image file %s error!\n", filename);
		return -1;
	}

	int magic = 0, num = 0, row = 0, col = 0;
	fread((uchar*)&magic, sizeof(magic), 1, fp);
	magic = decode(magic);
	fread((uchar*)&num, sizeof(num), 1, fp);
	num = decode(num);
	fread((uchar*)&row, sizeof(row), 1, fp);
	row = decode(row);
	fread((uchar*)&col, sizeof(col), 1, fp);
	col = decode(col);

	testcases = num;
	nSet = vector<mnistNode>(num);

	for (int i = 0; i < num; ++i) {
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				uchar tmp = 0;
				fread((uchar*)&tmp, sizeof(tmp), 1, fp);
				nSet[i].img[r + 2][c + 2] = (double)tmp / 255.0;
				// fprintf(stderr, "%.2f\n", nSet[i].img(r + 2, c + 2));
			}
		}
	}

	fclose(fp);
	return 0;
}

inline int readLabel(const char *filename, vector<mnistNode>& nSet) {
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		fprintf(stderr, "Open label file %s error!\n", filename);
		return -1;
	}

	int magic = 0, num = 0;
	fread((uchar*)&magic, sizeof(magic), 1, fp);
	magic = decode(magic);
	fread((uchar*)&num, sizeof(num), 1, fp);
	num = decode(num);

	for (int i = 0; i < num; ++i) {
		uchar tmp = 0;
		fread((uchar*)&tmp, sizeof(tmp), 1, fp);
		nSet[i].label[0][(int)tmp] = 1.0;
	}

	fclose(fp);
	return 0;
}

inline int readMnist(mnist& input) {
	if (readImage(__TRAIN_IMG__, input.trainCases, input.trainSet) == -1) return -1;
	if (readLabel(__TRAIN_LAB__, input.trainSet) == -1) return -1;
	if (readImage(__TEST_IMG__, input.testCases, input.testSet) == -1) return -1;
	if (readLabel(__TEST_LAB__, input.testSet) == -1) return -1;
	return 0;
}

inline const char *toString(const int& x) {
	int len = 0;
	if (x == 0) len = 1;
	else for (int tmp = x; tmp; ++len, tmp /= 10);
	char *s = new char[len + 1];
	if (x == 0) s[0] = '0';
	for (int tmp = x, nw = len; tmp; tmp /= 10) s[--nw] = tmp % 10 + '0';
	s[len] = 0;
	return s;
}

inline const char *cat(const char *s1, const char *s2) {
	int n = strlen(s1), m = strlen(s2), len = 0;
	char *s = new char[n + m + 1];
	for (int i = 0; i < n; ++i) s[len++] = s1[i];
	for (int i = 0; i < m; ++i) s[len++] = s2[i];
	s[len] = 0;
	return s;
}

inline int saveImage(const char *filename, const int& testcases, const vector<mnistNode>& nSet) {
	for (int i = 0; i < testcases; ++i) {
		const char *imagefile = cat(filename, cat(toString(i), ".bin"));
		FILE *fp = fopen(imagefile, "wb");
		if (fp == NULL) {
			fprintf(stderr, "Write image file %s failed!\n", imagefile);
			return -1;
		}

		for (int r = 0; r < 32; ++r) {
			for (int c = 0; c < 32; ++c) {
				fwrite(&nSet[i].img[r][c], sizeof(double), 1, fp);
			}
		}

		fclose(fp);
	}

	return 0;
}

#endif
#pragma once

#ifndef LENET_H
#define LENET_H

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include "matrix.h"
#include "readmnist.h"

const unsigned long long mt19937_max = 4294967293;

struct convolutionCell {
	Matrix W, dW;
	double B;
	convolutionCell() { B = 0.1; }
};

struct samplingCell {
	Matrix X, dW, Y;
	samplingCell() {}
};

struct fullConnectedCell {
	Matrix W;
	double B, delta, Y;
	fullConnectedCell() { B = 0.1, delta = Y = 0; }
};

struct convolutionLayer_1 {
	Matrix X;
	convolutionCell cells[6];
	convolutionLayer_1() {
		X = Matrix(32, vector<double>(32, 0.0));

		std::random_device e;
		std::mt19937 rng(e());
		
		for (int i = 0; i < 6; ++i) {
			cells[i].W = Matrix(5, vector<double>(5, 0.0));
			cells[i].dW = Matrix(28, vector<double>(28, 0.0));

			for (int r = 0; r < 5; ++r) {
				for (int c = 0; c < 5; ++c) {
					double rnd = (((double)rng() / (double)mt19937_max) - 0.5) * 2.0;
					cells[i].W[r][c] = rnd * sqrt(6.0 / (5.0 * 5.0 * (1.0 + 6.0)));
				}
			}
		}
	}
};

struct samplingLayer_1 {
	samplingCell cells[6];
	samplingLayer_1() {
		for (int i = 0; i < 6; ++i) {
			cells[i].X = Matrix(28, vector<double>(28, 0.0));
			cells[i].dW = Matrix(14, vector<double>(14, 0.0));
			cells[i].Y = Matrix(14, vector<double>(14, 0.0));
		}
	}
};

struct convolutionLayer_2 {
	convolutionCell cells[16];
	convolutionLayer_2() {
		std::random_device e;
		std::mt19937 rng(e());

		for (int i = 0; i < 16; ++i) {
			cells[i].W = Matrix(5, vector<double>(5, 0.0));
			cells[i].dW = Matrix(10, vector<double>(10, 0.0));

			for (int r = 0; r < 5; ++r) {
				for (int c = 0; c < 5; ++c) {
					double rnd = (((double)rng() / (double)mt19937_max) - 0.5) * 2.0;
					cells[i].W[r][c] = rnd * sqrt(6.0 / (5.0 * 5.0 * (6.0 + 16.0)));
				}
			}
		}
	}
};

struct samplingLayer_2 {
	samplingCell cells[16];
	samplingLayer_2() {
		for (int i = 0; i < 16; ++i) {
			cells[i].X = Matrix(10, vector<double>(10, 0.0));
			cells[i].dW = Matrix(5, vector<double>(5, 0.0));
			cells[i].Y = Matrix(5, vector<double>(5, 0.0));
		}
	}
};

struct outputLayer {
	Matrix V;
	fullConnectedCell cells[10];
	outputLayer() {
		V = Matrix(1, vector<double>(400, 0.0));

		std::random_device e;
		std::mt19937 rng(e());

		for (int i = 0; i < 10; ++i) {
			cells[i].W = Matrix(1, vector<double>(400, 0.0));

			for (int c = 0; c < 400; ++c) {
				double rnd = (((double)rng() / (double)mt19937_max) - 0.5) * 2.0;
				cells[i].W[0][c] = rnd * sqrt(6.0 / (400.0 + 10.0));
			}
		}
	}
};

struct LeNet_5 {
	convolutionLayer_1 C1;
	samplingLayer_1 S2;
	convolutionLayer_2 C3;
	samplingLayer_2 S4;
	outputLayer O5;
	LeNet_5() {}
};

const unsigned int map1[16][6] = {
	{1, 1, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 1, 1},
	{1, 0, 0, 0, 1, 1}, {1, 1, 0, 0, 0, 1}, {1, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 0},
	{0, 0, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 0, 0, 1, 1}, {1, 1, 1, 0, 0, 1},
	{1, 1, 0, 1, 1, 0}, {0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 1}
};

const unsigned int map2[6][16] = {
	{1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
	{1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
	{1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
	{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
	{0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
	{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1}
};

constexpr double max(const double& x, const double& y) { return x < y ? y : x; }
constexpr double ReLU(double z) { return z > 0 ? z : 0.0; }
constexpr double dReLU(double y) { return y > 0 ? 1.0 : 0.0; }

inline void forwardConvolutionLayer_1(const mnistNode& img, LeNet_5& net) {
	net.C1.X = img.img;

	for (int i = 0; i < 6; ++i) {
		
		for (int r = 0; r < 28; ++r) {
			for (int c = 0; c < 28; ++c) {
				Matrix ms = block(img.img, r, c, 5);
				double z = dot(ms, net.C1.cells[i].W);
				double y = ReLU(z);
				net.S2.cells[i].X[r][c] = y;
			}
		}
	}
}

inline void forwardSamplingLayer_1(LeNet_5& net) {
	for (int i = 0; i < 6; ++i) {	
		for (int r = 0; r < 14; ++r) {
			for (int c = 0; c < 14; ++c) {
				double pmax = -1e30;

				pmax = max(pmax, net.S2.cells[i].X[r << 1][c << 1]);
				pmax = max(pmax, net.S2.cells[i].X[r << 1][c << 1 | 1]);
				pmax = max(pmax, net.S2.cells[i].X[r << 1 | 1][c << 1]);
				pmax = max(pmax, net.S2.cells[i].X[r << 1 | 1][c << 1 | 1]);
				net.S2.cells[i].Y[r][c] = pmax;
			}
		}
	}
}

inline void forwardConvolutionLayer_2(LeNet_5& net) {
	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 10; ++r) {
			for (int c = 0; c < 10; ++c) {
				double sW = 0.0;

				for (int j = 0; j < 6; ++j) {
					if (!map1[i][j]) continue;

					Matrix ms = block(net.S2.cells[j].Y, r, c, 5);

					sW += dot(ms, net.C3.cells[i].W);
				}

				double z = sW + net.C3.cells[i].B;
				double y = ReLU(z);
				net.S4.cells[i].X[r][c] = y;
			}
		}
	}
}

inline void forwardSamplingLayer_2(LeNet_5& net) {
	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				double pmax = -1e30;

				pmax = max(pmax, net.S4.cells[i].X[r << 1][c << 1]);
				pmax = max(pmax, net.S4.cells[i].X[r << 1][c << 1 | 1]);
				pmax = max(pmax, net.S4.cells[i].X[r << 1 | 1][c << 1]);
				pmax = max(pmax, net.S4.cells[i].X[r << 1 | 1][c << 1 | 1]);
				net.S4.cells[i].Y[r][c] = pmax;
			}
		}
	}
}

inline void forwardOutputLayer(LeNet_5& net) {
	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				net.O5.V[0][i * 25 + r * 5 + c] = net.S4.cells[i].Y[r][c];
			}
		}
	}

	for (int i = 0; i < 10; ++i) {
		double z = dot(net.O5.cells[i].W, net.O5.V) + net.O5.cells[i].B;
		double y = ReLU(z);
		net.O5.cells[i].Y = y;
	}
}

inline void forwordPropagation(const mnistNode& img, LeNet_5& net) {
	forwardConvolutionLayer_1(img, net);
	forwardSamplingLayer_1(net);
	forwardConvolutionLayer_2(net);
	forwardSamplingLayer_2(net);
	forwardOutputLayer(net);
}

inline void backwardOutputLayer(const mnistNode& img, LeNet_5& net) {
	for (int i = 0; i < 10; ++i) {
		double e = net.O5.cells[i].Y - img.label[0][i];
		net.O5.cells[i].delta = e;
	}
}

inline void backwardSamplingLayer_2(LeNet_5& net) {
	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				for (int j = 0; j < 10; ++j) {
					net.S4.cells[i].dW[r][c] += net.O5.cells[j].W[0][i * 25 + r * 5 + c] * net.O5.cells[j].delta;
				}
			}
		}
	}
}

inline void backwardConvolutionLayer_2(LeNet_5& net) {
	for (int i = 0; i < 16; ++i) {
		Matrix delta = maxPadding(10, net.S4.cells[i].dW, 2, net.S4.cells[i].Y, net.S4.cells[i].X);
		
		for (int r = 0; r < 10; ++r) {
			for (int c = 0; c < 10; ++c) {
				net.C3.cells[i].dW[r][c] += delta[r][c] * dReLU(net.S4.cells[i].X[r][c]);
			}
		}
	}
}

inline void backwardSamplingLayer_1(LeNet_5& net) {
	Matrix tmpB[16];
	for (int i = 0; i < 16; ++i) tmpB[i] = Matrix(14, vector<double>(14, 0.0));

	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 10; ++r) {
			for (int c = 0; c < 10; ++c) {
				blockAdd(tmpB[i], r, c, net.C3.cells[i].W * net.C3.cells[i].dW[r][c]);
			}
		}
	}

	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 16; ++j) {
			if (!map2[i][j]) continue;

			net.S2.cells[i].dW = net.S2.cells[i].dW + tmpB[j];
		}
	}
}

inline void backwardConvolutionLayer_1(LeNet_5& net) {
	for (int i = 0; i < 6; ++i) {
		Matrix delta = maxPadding(28, net.S2.cells[i].dW, 2, net.S2.cells[i].Y, net.S2.cells[i].X);
		
		for (int r = 0; r < 28; ++r) {
			for (int c = 0; c < 28; ++c) {
				net.C1.cells[i].dW[r][c] += delta[r][c] * dReLU(net.S2.cells[i].X[r][c]);
			}
		}
	}
}

inline void backPropagation(const mnistNode& img, LeNet_5& net) {
	backwardOutputLayer(img, net);
	backwardSamplingLayer_2(net);
	backwardConvolutionLayer_2(net);
	backwardSamplingLayer_1(net);
	backwardConvolutionLayer_1(net);
}

inline void gradientDescendMethod(LeNet_5& net) {
	const double alpha = 0.01;

	for (int i = 0; i < 6; ++i) {
		Matrix dW = Matrix(5, vector<double>(5, 0.0));

		for (int r = 0; r < 28; ++r) {
			for (int c = 0; c < 28; ++c) {
				Matrix ms = block(net.C1.X, r, c, 5);

				dW = dW + ms * net.C1.cells[i].dW[r][c];
			}
		}

		dW = dW * -alpha;
		net.C1.cells[i].W = net.C1.cells[i].W + dW;
		net.C1.cells[i].B += -alpha * sum(net.C1.cells[i].dW);
	}

	for (int i = 0; i < 16; ++i) {
		Matrix X = Matrix(14, vector<double>(14, 0.0)), dW = Matrix(5, vector<double>(5, 0.0));

		for (int j = 0; j < 6; ++j) {
			if (!map1[i][j]) continue;

			X = X + net.S2.cells[j].Y;
		}

		for (int r = 0; r < 10; ++r) {
			for (int c = 0; c < 10; ++c) {
				Matrix ms = block(X, r, c, 5);

				dW = dW + ms * net.C3.cells[i].dW[r][c];
			}
		}

		dW = dW * -alpha;
		net.C3.cells[i].W = net.C3.cells[i].W + dW;
		net.C3.cells[i].B += -alpha * sum(net.C3.cells[i].dW);
	}

	for (int i = 0; i < 10; ++i) {
		net.O5.cells[i].W = net.O5.cells[i].W + net.O5.V * (-alpha * net.O5.cells[i].delta);
		net.O5.cells[i].B += -alpha * net.O5.cells[i].delta;
	}
}

/*
inline void reSet(LeNet_5& net) {
	for (int i = 0; i < 6; ++i) net.C1.cells[i].dW = Matrix(28, vector<double>(28, 0.0));
	for (int i = 0; i < 6; ++i) net.S2.cells[i].dW = Matrix(14, vector<double>(14, 0.0));
	for (int i = 0; i < 16; ++i) net.C3.cells[i].dW = Matrix(10, vector<double>(10, 0.0));
	for (int i = 0; i < 16; ++i) net.S4.cells[i].dW = Matrix(5, vector<double>(5, 0.0));
}
*/

inline void train(const mnist& input, LeNet_5& net) {
	for (int i = 0; i < input.trainCases; ++i) {
		forwordPropagation(input.trainSet[i], net);
		backPropagation(input.trainSet[i], net);
		gradientDescendMethod(net);
		reSet(net);
	}

	fprintf(stderr, "Training accomplished!\n");
}

/*
inline void trainBatch(const mnist& input, LeNet_5& net, int op, int batchCnt) {
	for (int i = 0; i < batchCnt; ++i) {
		forwordPropagation(input.trainSet[op + i], net);
		backPropagation(input.trainSet[op + i], net);
		gradientDescendMethod(net);
		reSet(net);
	}

	fprintf(stderr, "Training batch accomplished!\n");
}
*/

inline double recognize(const mnist& input, LeNet_5& net) {
	double accepted = 0.0;

	for (int i = 0; i < input.testCases; ++i) {
		double pmax = -1.0;
		int maxIndex = 0, stdIndex = 0;

		forwordPropagation(input.testSet[i], net);

		for (int j = 0; j < 10; ++j) {
			if (pmax < net.O5.cells[j].Y) pmax = net.O5.cells[j].Y, maxIndex = j;
		}
		for (int j = 0; j < 10; ++j) {
			if (input.testSet[i].label[0][j] == 1.0) {
				stdIndex = j;
				break;
			}
		}

		if (maxIndex == stdIndex) ++accepted;
		else fprintf(stderr, "read %d except %d.\n", maxIndex, stdIndex);
	}

	fprintf(stderr, "Accepted rate = %.6f\n", accepted / input.testCases);
	return accepted / input.testCases;
}

/*
inline double recognizeBatch(const mnist& input, LeNet_5& net) {
	const int batchCnt = 1000;
	double accepted = 0.0;

	for (int i = 0; i < batchCnt; ++i) {
		double pmax = -1.0;
		int maxIndex = 0, stdIndex = 0;

		forwordPropagation(input.testSet[i % 100 + (i / 100) * 100], net);

		for (int j = 0; j < 10; ++j) {
			// fprintf(stderr, "%.2f %.2f\n", net.O5.cells[j].Y, input.testSet[i % 100 + (i / 100) * 100].label[0][j]);
			if (pmax < net.O5.cells[j].Y) pmax = net.O5.cells[j].Y, maxIndex = j;
		}
		// fprintf(stderr, "===\n");
		for (int j = 0; j < 10; ++j) {
			if (input.testSet[i % 100 + i / 100 % 100].label[0][j] == 1.0) {
				stdIndex = j;
				break;
			}
		}

		if (maxIndex == stdIndex) ++accepted;
		else fprintf(stderr, "read %d except %d.\n", maxIndex, stdIndex);
	}
	fprintf(stderr, "Accepted rate = %.6f\n", accepted / batchCnt);
	return accepted / batchCnt;
}

inline int save(const LeNet_5& net) {
	FILE *fp = fopen("well_trained.txt", "w");

	if (fp == NULL) {
		fprintf(stderr, "saving file error!\n");
		return -1;
	}

	for (int i = 0; i < 6; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				fprintf(fp, "%.15f%c", net.C1.cells[i].W[r][c], c == 4 ? '\n' : ' ');
			}
		}

		fprintf(fp, "%.15\n", net.C1.cells[i].B);
	}

	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				fprintf(fp, "%.15f%c", net.C3.cells[i].W[r][c], c == 4 ? '\n' : ' ');
			}
		}

		fprintf(fp, "%.15\n", net.C3.cells[i].B);
	}

	for (int i = 0; i < 10; ++i) {
		for (int c = 0; c < 400; ++c) {
			fprintf(fp, "%.15f%c", net.O5.cells[i].W[0][c], c == 4 ? '\n' : ' ');
		}

		fprintf(fp, "%.15\n", net.O5.cells[i].B);
	}

	return 0;
}

inline int load(LeNet_5& net) {
	FILE *fp = fopen("well_trained.txt", "r");

	if (fp == NULL) {
		fprintf(stderr, "load file error!\n");
		return -1;
	}

	for (int i = 0; i < 6; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				fscanf(fp, "%lf", &net.C1.cells[i].W[r][c]);
			}
		}

		fscanf(fp, "%lf", &net.C1.cells[i].B);
	}

	for (int i = 0; i < 16; ++i) {
		for (int r = 0; r < 5; ++r) {
			for (int c = 0; c < 5; ++c) {
				fscanf(fp, "%lf", &net.C3.cells[i].W[r][c]);
			}
		}

		fscanf(fp, "%lf", &net.C3.cells[i].B);
	}

	for (int i = 0; i < 10; ++i) {
		for (int c = 0; c < 400; ++c) {
			fscanf(fp, "%lf", &net.O5.cells[i].W[0][c]);
		}

		fscanf(fp, "%lf", &net.O5.cells[i].B);
	}

	return 0;
}
*/

#endif
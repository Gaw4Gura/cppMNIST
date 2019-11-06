#include "lenet.h"
#include "readmnist.h"

LeNet_5 net;
mnist input;

int main() {
	readMnist(input);

	train(input, net);

	// load(net);

	double accracy = recognize(input, net);

	printf("Accuracy: %.3f\n", accracy);
	
	// save(net);

	system("pause");

	return 0;
}
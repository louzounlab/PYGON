#include "../includes/MathUtils.h"

float MathUtils::calculateStd(const std::vector<float>& data) {
	float standardDeviation = 0.0f;
	int len = data.size();
	int nonZero = 0;
	float mean = calculateMeanWithoutZeroes(data);
	for (int i = 0; i < len; i++)
		if (data[i] != 0) {
			nonZero++;
			standardDeviation += (data[i] - mean) * (data[i] - mean);
		}

	standardDeviation = sqrt(standardDeviation / ((float) nonZero));

	return standardDeviation;
}

float MathUtils::calculateMean(const std::vector<float>& data) {
	int len = data.size();
	float sum = 0.0f;
	for (int i = 0; i < len; i++) {
		sum += data[i];
	}
	return sum / ((float) len);
}

float MathUtils::calculateMeanWithoutZeroes(const std::vector<float>& data) {

	int len = data.size();
	int nonZero = 0;
	float sum = 0.0f;
	for (int i = 0; i < len; i++) {
		if (data[i] != 0) {
			sum += data[i];
			nonZero++;
		}
	}
	return sum / ((float) nonZero);

}

float MathUtils::calculateWeightedAverage(const std::vector<float> &data,
		const std::vector<float> &weights, int sizes) {

	float sum = 0.0f;
	for (int i = 0; i < sizes; i++)
		sum += data[i] * weights[i];

	float weightSum = 0;
	for (int i = 0; i < sizes; i++)
		weightSum += weights[i];

	sum = sum / weightSum;
	return sum;
}

float MathUtils::calculateWeightedStd(const std::vector<float>& data,
		const std::vector<float>& weights, int sizes) {

	float avg = calculateWeightedAverage(data ,weights, sizes);
	std::vector<float> modified_data(sizes);
	for(int i = 0; i < sizes; i++)
		modified_data[i] = (data[i]-avg)*(data[i]-avg);
	float variance = calculateWeightedAverage(modified_data, weights, sizes);
	return sqrt(variance);
}

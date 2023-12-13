#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

template <typename T>
__global__ void binary_cross_entropy_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T * __restrict__ predictions,
	const float * __restrict__ targets,
	float * __restrict__ values,
	T * __restrict__ gradients)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;

	const uint32_t dim_idx = i % stride;
	const uint32_t elem_idx = i / stride;
	if (dim_idx >= dims)
	{
		values[i] = 0;
		gradients[i] = 0;
		return;
	}

	const uint32_t target_idx = elem_idx * dims + dim_idx;

	const uint32_t n_total = n_elements / stride * dims;

	const float epsilon = 1e-5;
    const float target = targets[target_idx];
	const float prediction = min(max((float)predictions[i],epsilon), 1.f - epsilon);

    values[i] = -(target * log(prediction) + (1.0 - target) * log(1.0 - prediction)) / n_total;

    float gradient = -(target-prediction) / (prediction - (prediction*prediction));
	gradients[i] = (T)(loss_scale * gradient / n_total);
}

template <typename T>
class BinaryCrossEntropyLoss : public tcnn::Loss<T>
{
public:
	void evaluate(
		cudaStream_t stream,
		const uint32_t stride,
		const uint32_t dims,
		const float loss_scale,
		const tcnn::GPUMatrix<T> & prediction,
		const tcnn::GPUMatrix<float> & target,
		tcnn::GPUMatrix<float> & values,
		tcnn::GPUMatrix<T> & gradients,
		const tcnn::GPUMatrix<float> * data_pdf = nullptr) const override
	{
		(void)data_pdf;

		if (prediction.n() != target.n())
		{
			throw std::runtime_error(
				std::string("Prediction and target don't have matching batch size ") + std::to_string(prediction.n())
				+ "!=" + std::to_string(target.n()));
		}

		if (prediction.m() != stride)
		{
			throw std::runtime_error(
				std::string("Prediction does not have appropriate dimensions ") + std::to_string(prediction.m())
				+ "!=" + std::to_string(stride));
		}

		if (target.m() != dims)
		{
			throw std::runtime_error(
				std::string("Target does not have appropriate dimensions ") + std::to_string(target.m())
				+ "!=" + std::to_string(dims));
		}

		tcnn::linear_kernel(
			binary_cross_entropy_loss<T>,
			0,
			stream,
			prediction.n_elements(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data());
	}

	void update_hyperparams(const nlohmann::json & params) override
	{
	}

	nlohmann::json hyperparams() const override
	{
		return {
			{"otype", "BinaryCrossEntropy"},
		};
	}
};
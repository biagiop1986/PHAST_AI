/* Copyright 2021 Biagio Peccerillo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __PHAST_AI_H
#define __PHAST_AI_H

#if defined(_PHAST_USING_CUDA)
#include "header_CUDA/ai_CUDA.h"
namespace __phast_internal_namespace = phast::ai::cuda;
#elif defined(_PHAST_USING_MULTI_CORE)
#include "header_MULTI/ai_MULTI.h"
namespace __phast_internal_namespace = phast::ai::multi;
#else
#error "One between _PHAST_USING_CUDA and _PHAST_USING_MULTI_CORE should be defined"
#endif

namespace phast
{
namespace ai
{

template <typename T>
void batch_convolution(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, const phast::vector<T>* bias, phast::cube<T>& output)
{
	const int padding = 0;

	if((num_filters == 0) || (stride <= 0))
	{
		output.assign(0, 0, 0); // silent fail or exception?
		return;
	}

	const int input_channels = filters.size_i() / num_filters;
	const int batch_size = images.size_i() / input_channels;
	
	// Other possible checks:
	// - all the filters should have the same dimensions
	// - filters should be smaller than the input image
	// - WE ASSUME THAT the number of channels is the same for filters and images
	// - the number of bias terms should be the same as the number of filters
	const int out_size_i = num_filters * batch_size;
	const int out_size_j = 1 + (images.size_j() - filters.size_j() + 2*padding) / stride;
	const int out_size_k = 1 + (images.size_k() - filters.size_k() + 2*padding) / stride;
	if((output.size_i() != out_size_i) || (output.size_j() != out_size_j) || (output.size_k() != out_size_k))
		output.assign(out_size_i, out_size_j, out_size_k);

	__phast_internal_namespace::batch_convolution(images, filters, num_filters, stride, batch_size, bias, output);
}

template <typename T>
void batch_convolution_channel_major(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, const phast::vector<T>* bias, phast::cube<T>& output)
{
	const int padding = 0;

	if((num_filters == 0) || (stride <= 0))
	{
		output.assign(0, 0, 0); // silent fail or exception?
		return;
	}

	const int input_channels = filters.size_i() / num_filters;
	const int batch_size = images.size_i() / input_channels;
	
	// Other possible checks:
	// - all the filters should have the same dimensions
	// - filters should be smaller than the input image
	// - WE ASSUME THAT the number of channels is the same for filters and images
	// - the number of bias terms should be the same as the number of filters
	const int out_size_i = num_filters * batch_size;
	const int out_size_j = 1 + (images.size_j() - filters.size_j() + 2*padding) / stride;
	const int out_size_k = 1 + (images.size_k() - filters.size_k() + 2*padding) / stride;
	if((output.size_i() != out_size_i) || (output.size_j() != out_size_j) || (output.size_k() != out_size_k))
		output.assign(out_size_i, out_size_j, out_size_k);

	__phast_internal_namespace::batch_convolution_channel_major(images, filters, num_filters, stride, batch_size, bias, output);
}

template <typename T>
phast::cube<T> batch_convolution(const phast::cube<T>& image, const phast::cube<T>& filters, int num_filters, int stride, const phast::vector<T>* bias)
{
	phast::cube<T> output;
	batch_convolution(image, filters, num_filters, stride, bias, output);
	return output;
}

template <typename T>
phast::cube<T> batch_convolution_channel_major(const phast::cube<T>& image, const phast::cube<T>& filters, int num_filters, int stride, const phast::vector<T>* bias)
{
	phast::cube<T> output;
	batch_convolution_channel_major(image, filters, num_filters, stride, bias, output);
	return output;
}

template <typename T>
void rotate_and_pad(const phast::cube<T>& src, const int ph, const int pw, phast::cube<T>& dst)
{
	const int HH = (ph << 1) + src.size_j();
	const int WW = (pw << 1) + src.size_k();

	if((dst.size_i() != src.size_i()) || (dst.size_j() != HH) || (dst.size_k() != WW))
		dst.assign(src.size_i(), (ph << 1) + src.size_j(), (pw << 1) + src.size_k(), phast::get_zero<T>::get());
	__phast_internal_namespace::rotate_and_pad(src, ph, pw, HH, WW, dst);
}

template <typename T>
phast::cube<T> rotate_and_pad(const phast::cube<T>& src, const int ph, const int pw)
{
	phast::cube<T> dst;
	rotate_and_pad(src, ph, pw, dst);
	return dst;
}

template <typename T>
void pad(const phast::cube<T>& src, const int ph, const int pw, phast::cube<T>& dst)
{
	const int HH = (ph << 1) + src.size_j();
	const int WW = (pw << 1) + src.size_k();

	if((dst.size_i() != src.size_i()) || (dst.size_j() != HH) || (dst.size_k() != WW))
		dst.assign(src.size_i(), (ph << 1) + src.size_j(), (pw << 1) + src.size_k(), phast::get_zero<T>::get());
	__phast_internal_namespace::pad(src, ph, pw, HH, WW, dst);
}

template <typename T>
phast::cube<T> pad(const phast::cube<T>& src, const int ph, const int pw)
{
	phast::cube<T> dst;
	pad(src, ph, pw, dst);
	return dst;
}

#if __cplusplus >= 201300
template <typename T>
auto const convolution = batch_convolution<T>;
template <typename T>
auto const convolution_channel_major = batch_convolution_channel_major<T>;
#else
template <typename T>
void convolution(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, const phast::vector<T>* bias, phast::cube<T>& output)
{
	return batch_convolution(images, filters, num_filters, stride, bias, output);
}
template <typename T>
void convolution_channel_major(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, const phast::vector<T>* bias, phast::cube<T>& output)
{
	return batch_convolution_channel_major(images, filters, num_filters, stride, bias, output);
}
#endif

}
}

#endif /* __PHAST_AI_H */

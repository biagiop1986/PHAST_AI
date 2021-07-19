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

#ifndef __AI_CUDA_H
#define __AI_CUDA_H

#include <phast.h>

namespace phast
{
namespace ai
{
namespace cuda
{

const int MAX_MANAGED_Hk = 32;

inline __host__ __device__ int iDivUp(int a, int b)
{
	return (a + b - 1) / b;
}

constexpr int __PHAST_MAX_TEXTURE_SIZE = 131072;
constexpr int __PHAST_TEXTURE_ALIGNMENT = 512; // bytes

template <typename T>
struct is_texture_type
{
    static const bool value = std::is_same<T, float>::value;
};

template <typename T>
bool is_texture_eligible(const T* ptr, int n_elems, typename std::enable_if<is_texture_type<T>::value>::type* = nullptr)
{
    return (n_elems * sizeof(T) <= __PHAST_MAX_TEXTURE_SIZE) && !((__PHAST_TEXTURE_ALIGNMENT-1) & (uintptr_t)ptr);
}

template <typename T>
bool is_texture_eligible(const T*, int, typename std::enable_if<!is_texture_type<T>::value>::type* = nullptr)
{
    return false;
}

template <typename T, typename Enable = void>
class image_wrapper {};

template <typename T>
class image_wrapper<T, typename std::enable_if<is_texture_type<T>::value>::type>
{
public:
    __host__ image_wrapper(const T* ptr, int n_elems)
    {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = (void*)ptr;
        resDesc.res.linear.sizeInBytes = n_elems * sizeof(T);
        resDesc.res.linear.desc = cudaCreateChannelDesc<T>();

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;

        cudaCreateTextureObject(&image, &resDesc, &texDesc, nullptr);
		base = 0;
    }
    __host__ void destroy()
    {
        cudaDestroyTextureObject(image);
    }
    __device__ __forceinline__ T operator[](int index) const
    {
        return tex1Dfetch<T>(image, base + index);
    }
	__device__ __forceinline__ void operator+=(int n)
	{
		base += n;
	}
	__device__ __forceinline__ void operator-=(int n)
	{
		base -= n;
	}
private:
    cudaTextureObject_t image;
	int base;
};

template <typename T>
class image_wrapper<T, typename std::enable_if<!is_texture_type<T>::value>::type>
{
public:
    __host__ image_wrapper(const T* ptr, int n_elems) : image(ptr) {}
    __host__ void destroy() {}
    __device__ __forceinline__ T operator[](int index) const
    {
        return image[base + index];
    }
	__device__ __forceinline__ void operator+=(int n)
	{
		base += n;
	}
	__device__ __forceinline__ void operator-=(int n)
	{
		base -= n;
	}
private:
    const T* image;
	int base;
};

template <typename T>
struct host_image_wrapper
{
    host_image_wrapper(const T* ptr, int n_elems) : image(ptr, n_elems) {}
    ~host_image_wrapper()
    {
        image.destroy();
    }
	operator image_wrapper<T>()
	{
		return image;
	}
	operator const image_wrapper<T>() const
	{
		return image;
	}

    image_wrapper<T> image;
};

template <typename T>
inline void adjust_block_size(int& block_size_x, int& block_size_y, int& smem_size, int Hk, int Wk, int D, int stride, int& hk, int& wk, 
	const T* filters, int num_filters)
{
    block_size_x = phast::custom::cuda::get_major_block_size();
	block_size_y = phast::custom::cuda::get_minor_block_size();

	if(block_size_x & (_PHAST_CUDA_WARP_SIZE - 1) != 0)
		block_size_x -= (block_size_x & (_PHAST_CUDA_WARP_SIZE - 1));
	if(block_size_x < _PHAST_CUDA_WARP_SIZE)
		block_size_x = _PHAST_CUDA_WARP_SIZE;
	if(block_size_x > _PHAST_CUDA_MAX_THREAD_PER_BLOCK)
		block_size_x = _PHAST_CUDA_MAX_THREAD_PER_BLOCK;

	if(block_size_y > D)
		block_size_y = D;

	hk = Hk; wk = Wk;
	if(hk > MAX_MANAGED_Hk)
		hk = MAX_MANAGED_Hk;
	if(wk > _PHAST_CUDA_WARP_SIZE)
		wk = _PHAST_CUDA_WARP_SIZE;

	if(!is_texture_eligible(filters, num_filters*D*Hk*Wk))
	{
		if((Wk == wk) && (Hk == hk)) // small filter
		{
			smem_size = Hk*Wk*sizeof(T);
		}
		else if (Wk == wk)
		{
			smem_size = hk*Wk*sizeof(T);
		}
		else // big filter
		{
			if(block_size_x < Wk)
			{
				block_size_x = Wk;
				if((block_size_x & (_PHAST_CUDA_WARP_SIZE - 1)) != 0)
					block_size_x += _PHAST_CUDA_WARP_SIZE - (block_size_x & (_PHAST_CUDA_WARP_SIZE - 1));
			}
			smem_size = hk*Wk*sizeof(T) + (1 + (Wk % _PHAST_CUDA_WARP_SIZE)) * (block_size_x / _PHAST_CUDA_WARP_SIZE - 1) * sizeof(T);
		}
	}
	else // no need to allocate shared memory for the filters!
	{
		if(Wk == wk)
		{
			smem_size = 1;
		}
		else
		{
			if(block_size_x < Wk)
			{
				block_size_x = Wk;
				if((block_size_x & (_PHAST_CUDA_WARP_SIZE - 1)) != 0)
					block_size_x += _PHAST_CUDA_WARP_SIZE - (block_size_x & (_PHAST_CUDA_WARP_SIZE - 1));
			}
			smem_size = (1 + (Wk % _PHAST_CUDA_WARP_SIZE)) * (block_size_x / _PHAST_CUDA_WARP_SIZE - 1) * sizeof(T);
		}
	}

	if(block_size_x > _PHAST_CUDA_MAX_THREAD_PER_BLOCK)
		throw std::runtime_error("This filter cannot fit into a single CUDA block");
	if(block_size_x * block_size_y > _PHAST_CUDA_MAX_THREAD_PER_BLOCK)
		block_size_y = _PHAST_CUDA_MAX_THREAD_PER_BLOCK / block_size_x;

	const int base_smem_size = smem_size;
	if(base_smem_size > phast::custom::cuda::get_total_shared())
        throw std::runtime_error("This filter cannot fit in shared memory - data-type is too big");

	smem_size = block_size_y * base_smem_size;
	if(smem_size > phast::custom::cuda::get_total_shared())
	{
		block_size_y = iDivUp(phast::custom::cuda::get_total_shared(), base_smem_size);
		smem_size = block_size_y * base_smem_size;
	}
}

template <typename T>
inline void adjust_block_size(int& block_size_x, int& block_size_y, int& smem_size, int Hk, int Wk, int D, int stride, int& hk, int& wk)
{
    block_size_x = phast::custom::cuda::get_major_block_size();
	block_size_y = phast::custom::cuda::get_minor_block_size();

	if(block_size_x & (_PHAST_CUDA_WARP_SIZE - 1) != 0)
		block_size_x -= (block_size_x & (_PHAST_CUDA_WARP_SIZE - 1));
	if(block_size_x < _PHAST_CUDA_WARP_SIZE)
		block_size_x = _PHAST_CUDA_WARP_SIZE;
	if(block_size_x > _PHAST_CUDA_MAX_THREAD_PER_BLOCK)
		block_size_x = _PHAST_CUDA_MAX_THREAD_PER_BLOCK;

	if(block_size_y > D)
		block_size_y = D;

	hk = Hk; wk = Wk;
	if(hk > MAX_MANAGED_Hk)
		hk = MAX_MANAGED_Hk;
	if(wk > _PHAST_CUDA_WARP_SIZE)
		wk = _PHAST_CUDA_WARP_SIZE;

	if((Wk == wk) && (Hk == hk)) // small filter
	{
		smem_size = Hk*Wk*sizeof(T);
	}
	else if (Wk == wk)
	{
		smem_size = hk*Wk*sizeof(T);
	}
	else // big filter
	{
		if(block_size_x < Wk)
		{
			block_size_x = Wk;
			if((block_size_x & (_PHAST_CUDA_WARP_SIZE - 1)) != 0)
				block_size_x += _PHAST_CUDA_WARP_SIZE - (block_size_x & (_PHAST_CUDA_WARP_SIZE - 1));
		}
		smem_size = hk*Wk*sizeof(T) + (1 + (Wk % _PHAST_CUDA_WARP_SIZE)) * (block_size_x / _PHAST_CUDA_WARP_SIZE - 1) * sizeof(T);
	}

	if(block_size_x > _PHAST_CUDA_MAX_THREAD_PER_BLOCK)
		throw std::runtime_error("This filter cannot fit into a single CUDA block");
	if(block_size_x * block_size_y > (_PHAST_CUDA_MAX_THREAD_PER_BLOCK >> 1))
		block_size_y = (_PHAST_CUDA_MAX_THREAD_PER_BLOCK >> 1) / block_size_x;

	const int base_smem_size = smem_size;
	if(base_smem_size > phast::custom::cuda::get_total_shared())
        throw std::runtime_error("This filter cannot fit in shared memory - data-type is too big");

	smem_size = block_size_y * base_smem_size;
	if(smem_size > phast::custom::cuda::get_total_shared())
	{
		block_size_y = iDivUp(phast::custom::cuda::get_total_shared(), base_smem_size);
		smem_size = block_size_y * base_smem_size;
	}
}

// BATCH CONVOLUTION
template <typename T, int Hk>
__global__ void batch_convolution_kernel(const T* image, const int Hi, const int Wi, const int D, const T* filters, const int num_filters, 
	const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int batch_index = blockIdx.z / num_filters;
	const int filter_index = blockIdx.z % num_filters;
	filters += filter_index*D*Hk*Wk;
	image += batch_index * D * Hi * Wi;

	T data[Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();
	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	int D_div_up = blockDim.y * iDivUp(D, blockDim.y);
	for(int d = threadIdx.y; d < D_div_up; d += blockDim.y)
	{
		if(d < D)
		{
			for(int i = threadIdx.x; i < Hk*Wk; i += blockDim.x)
				shared[i] = filters[d*Hk*Wk + i];
		}
		__syncthreads();

		if(d < D)
		{
			int index = d * Hi * Wi + tidy * Wi + tidx;
	
			for(int s = 0; s < Hk; ++s)
			{
				int tidy_ = tidy + s;
				if(tidx < Wi && tidy_ < Hi)
					data[s] = image[index];
				else
					data[s] = phast::get_zero<T>::get();
				index += Wi;
			}
	
			T sum = phast::get_zero<T>::get();
	
			for(int m = 0; m < Wk; ++m)
			{
				if(m > 0)
					sum = __shfl_up_sync(0xffffffff, sum, 1);
	
//				#pragma unroll
				for(int n = 0; n < Hk; ++n)
				{
					sum = data[n] * shared[n*Wk + m] + sum;
				}
			}
	
			total_sum += sum;
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_kernel_bigHk_filter(const T* image, const int Hi, const int Wi, const int D, const T* filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int batch_index = blockIdx.z / num_filters;
	const int filter_index = blockIdx.z % num_filters;
	filters += filter_index*D*Hk*Wk;
	image += batch_index * D * Hi * Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();
	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk, tidy += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		int D_div_up = blockDim.y * iDivUp(D, blockDim.y);
		for(int d = threadIdx.y; d < D_div_up; d += blockDim.y)
		{
			if(d < D)
			{
				for(int i = threadIdx.x; i < h_range*Wk; i += blockDim.x)
					shared[i] = filters[d*Hk*Wk + h1*Wk + i];
			}
			__syncthreads();

			if(d < D)
			{	
				int index = d * Hi * Wi + tidy * Wi + tidx;
		
				for(int s = 0; s < h_range; ++s)
				{
					int tidy_ = tidy + s;
					if(tidx < Wi && tidy_ < Hi)
						data[s] = image[index];
					else
						data[s] = phast::get_zero<T>::get();
					index += Wi;
				}
		
				T sum = phast::get_zero<T>::get();
		
				for(int m = 0; m < Wk; ++m)
				{
					if(m > 0)
						sum = __shfl_up_sync(0xffffffff, sum, 1);
		
//					#pragma unroll
					for(int n = 0; n < h_range; ++n)
					{
						sum = data[n] * shared[n*Wk + m] + sum;
					}
				}
		
				total_sum += sum;
			}
			__syncthreads();
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_kernel_big_filter(const T* image, const int Hi, const int Wi, const int D, const T* filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int BLOCK_PROCESS_DATA_COUNT = (blockDim.x - Wk + 1) / stride;
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;

	int THIS_BLOCK_PROCESS_DATA_COUNT;
	if((blockIdx.x + 1)*BLOCK_PROCESS_DATA_COUNT <= Wo)
		THIS_BLOCK_PROCESS_DATA_COUNT = BLOCK_PROCESS_DATA_COUNT;
	else
		THIS_BLOCK_PROCESS_DATA_COUNT = Wo - blockIdx.x*BLOCK_PROCESS_DATA_COUNT;

	const int batch_index = blockIdx.z / num_filters;
	const int filter_index = blockIdx.z % num_filters;
	filters += filter_index*D*Hk*Wk;
	image += batch_index*D*Hi*Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	int tidx, tidy;

	T* shared = SharedMemory<T>();
	T* prov = shared + RED_Hk*Wk;
	for(int i = threadIdx.x; i < (WARP_COUNT - 1)*THIS_BLOCK_PROCESS_DATA_COUNT; i += blockDim.x)
		prov[i] = phast::get_zero<T>::get();
	__syncthreads();

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + threadIdx.x;
		tidy = stride*blockIdx.y + h1;

		int D_div_up = blockDim.y * iDivUp(D, blockDim.y);
		for(int d = threadIdx.y; d < D_div_up; d += blockDim.y)
		{
			if(d < D)
			{
				for(int i = threadIdx.x; i < h_range*Wk; i += blockDim.x)
					shared[i] = filters[d*Hk*Wk + h1*Wk + i];
			}
			__syncthreads();

			if(d < D)
			{
				int index = d * Hi * Wi + tidy * Wi + tidx;
	
				for(int s = 0; s < h_range; ++s)
				{
					int tidy_ = tidy + s;
					if(tidx < Wi && tidy_ < Hi)
						data[s] = image[index];
					else
						data[s] = phast::get_zero<T>::get();
					index += Wi;
				}
	
				T sum = phast::get_zero<T>::get();
	
				int p = THIS_BLOCK_PROCESS_DATA_COUNT;
				if(warpId == 0)
				{
					for(int m = 0; m < _PHAST_CUDA_WARP_SIZE; ++m)
					{
						if(m > 0)
							sum = __shfl_up_sync(0xffffffff, sum, 1);
	
						for(int n = 0; n < h_range; ++n)
							sum = data[n] * shared[n*Wk + m] + sum;
	
			            if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
			                prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
					}
				}
				else if(warpId != (WARP_COUNT -1))
				{
					const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
					for(int m = off; m < _PHAST_CUDA_WARP_SIZE; ++m)
					{
						if(m > off)
							sum = __shfl_up_sync(0xffffffff, sum, 1);
	
						if(laneId == 0)
							sum = phast::get_zero<T>::get();
	
						for(int n = 0; n < h_range; ++n)
							sum = data[n] * shared[n*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;
	
			            if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
			                prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
					}
				}
				else
				{
					const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
					for(int m = off; m < (Wk % _PHAST_CUDA_WARP_SIZE); ++m)
					{
						if(m > off)
							sum = __shfl_up_sync(0xffffffff, sum, 1);
						
						if(laneId == 0)
							sum = phast::get_zero<T>::get();
	
						for(int n = 0; n < h_range; ++n)
							sum = data[n] * shared[n*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;
					}
				}
	
				total_sum += sum;
			}
			__syncthreads();
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	int small_tidx = (threadIdx.x - Wk + 1) / stride;
	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + small_tidx;
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
    if((warpId == (WARP_COUNT - 1)) && (small_tidx >= 0) && (small_tidx < THIS_BLOCK_PROCESS_DATA_COUNT) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
	{
        for(int k = 0; k < (WARP_COUNT - 1); ++k)
            total_sum += prov[k*THIS_BLOCK_PROCESS_DATA_COUNT + small_tidx];

        output[index] = total_sum;
	}
}

template <typename T, int Hk>
__global__ void batch_convolution_cm_kernel(const T* image, const int Hi, const int Wi, const int D, const T* filters, const int num_filters, 
	const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int filter_index = blockIdx.z / batch_size;
	const int batch_index = blockIdx.z % batch_size;
	filters += filter_index*Hk*Wk;
	image += batch_index*Hi*Wi;

	T data[Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();

	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	int D_div_up = blockDim.y * iDivUp(D, blockDim.y);
	for(int d = threadIdx.y; d < D_div_up; d += blockDim.y)
	{
		if(d < D)
		{
			for(int i = threadIdx.x; i < Hk*Wk; i += blockDim.x)
				shared[i] = filters[d*num_filters*Hk*Wk + i];
		}
		__syncthreads();

		if(d < D)
		{
			int index = d * batch_size * Hi * Wi + tidy * Wi + tidx;

			for(int s = 0; s < Hk; ++s)
			{
				int tidy_ = tidy + s;
				if(tidx < Wi && tidy_ < Hi)
					data[s] = image[index];
				else
					data[s] = phast::get_zero<T>::get();
				index += Wi;
			}

			T sum = phast::get_zero<T>::get();

			for(int m = 0; m < Wk; ++m)
			{
				if(m > 0)
					sum = __shfl_up_sync(0xffffffff, sum, 1);

//				#pragma unroll
				for(int n = 0; n < Hk; ++n)
				{
					sum = data[n] * shared[n*Wk + m] + sum;
				}
			}

			total_sum += sum;
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_cm_kernel_bigHk_filter(const T* image, const int Hi, const int Wi, const int D, const T* filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int filter_index = blockIdx.z / batch_size;
	const int batch_index = blockIdx.z % batch_size;
	filters += filter_index*Hk*Wk;
	image += batch_index*Hi*Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();

	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk, tidy += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		int D_div_up = blockDim.y * iDivUp(D, blockDim.y);
		for(int d = threadIdx.y; d < D_div_up; d += blockDim.y)
		{
			if(d < D)
			{
				for(int i = threadIdx.x; i < h_range*Wk; i += blockDim.x)
					shared[i] = filters[d*num_filters*Hk*Wk + h1*Wk + i];
			}
			__syncthreads();

			if(d < D)
			{
				int index = d * batch_size * Hi * Wi + tidy * Wi + tidx;
	
				for(int s = 0; s < h_range; ++s)
				{
					int tidy_ = tidy + s;
					if(tidx < Wi && tidy_ < Hi)
						data[s] = image[index];
					else
						data[s] = phast::get_zero<T>::get();
					index += Wi;
				}
	
				T sum = phast::get_zero<T>::get();
	
				for(int m = 0; m < Wk; ++m)
				{
					if(m > 0)
						sum = __shfl_up_sync(0xffffffff, sum, 1);
	
//					#pragma unroll
					for(int n = 0; n < h_range; ++n)
					{
						sum = data[n] * shared[n*Wk + m] + sum;
					}
				}
	
				total_sum += sum;
			}
			__syncthreads();
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_cm_kernel_big_filter(const T* image, const int Hi, const int Wi, const int D, const T* filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int BLOCK_PROCESS_DATA_COUNT = (blockDim.x - Wk + 1) / stride;
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;

    int THIS_BLOCK_PROCESS_DATA_COUNT;
    if((blockIdx.x + 1)*BLOCK_PROCESS_DATA_COUNT <= Wo)
        THIS_BLOCK_PROCESS_DATA_COUNT = BLOCK_PROCESS_DATA_COUNT;
    else
        THIS_BLOCK_PROCESS_DATA_COUNT = Wo - blockIdx.x*BLOCK_PROCESS_DATA_COUNT;

	const int filter_index = blockIdx.z /batch_size;
	const int batch_index = blockIdx.z % batch_size;
	filters += filter_index*Hk*Wk;
	image += batch_index*Hi*Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	int tidx, tidy;

	T* shared = SharedMemory<T>();
	T* prov = shared + RED_Hk*Wk;
	for(int i = threadIdx.x; i < (WARP_COUNT - 1)*BLOCK_PROCESS_DATA_COUNT; i += blockDim.x)
		prov[i] = phast::get_zero<T>::get();
	__syncthreads();

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + threadIdx.x;
		tidy = stride*blockIdx.y + h1;

		int D_div_up = blockDim.y * iDivUp(D, blockDim.y);
		for(int d = threadIdx.y; d < D_div_up; d += blockDim.y)
		{
			if(d < D)
			{
				for(int i = threadIdx.x; i < h_range*Wk; i += blockDim.x)
					shared[i] = filters[d*num_filters*Hk*Wk + h1*Wk + i];
			}
			__syncthreads();

			if(d < D)
			{
				int index = d * batch_size * Hi * Wi + tidy * Wi + tidx;
	
				for(int s = 0; s < h_range; ++s)
				{
					int tidy_ = tidy + s;
					if(tidx < Wi && tidy_ < Hi)
						data[s] = image[index];
					else
						data[s] = phast::get_zero<T>::get();
					index += Wi;
				}
	
				T sum = phast::get_zero<T>::get();
	
				int p = THIS_BLOCK_PROCESS_DATA_COUNT;
	            if(warpId == 0)
	            {
	                for(int m = 0; m < _PHAST_CUDA_WARP_SIZE; ++m)
	                {
	                    if(m > 0)
	                        sum = __shfl_up_sync(0xffffffff, sum, 1);
	
	                    for(int n = 0; n < h_range; ++n)
	                        sum = data[n] * shared[n*Wk + m] + sum;
	
	                    if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
	                        prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
	                }
	            }
	            else if(warpId != (WARP_COUNT -1))
	            {
	                const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
	                for(int m = off; m < _PHAST_CUDA_WARP_SIZE; ++m)
	                {
	                    if(m > off)
	                        sum = __shfl_up_sync(0xffffffff, sum, 1);
	
	                    if(laneId == 0)
	                        sum = phast::get_zero<T>::get();
	
	                    for(int n = 0; n < h_range; ++n)
	                        sum = data[n] * shared[n*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;
	
	                    if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
	                        prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
	                }
	            }
	            else
	            {
	                const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
	                for(int m = off; m < (Wk % _PHAST_CUDA_WARP_SIZE); ++m)
	                {
	                    if(m > off)
	                        sum = __shfl_up_sync(0xffffffff, sum, 1);
	
	                    if(laneId == 0)
	                        sum = phast::get_zero<T>::get();
	
	                    for(int n = 0; n < h_range; ++n)
	                        sum = data[n] * shared[n*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;
	                }
	            }
	
				total_sum += sum;
			}
			__syncthreads();
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	int small_tidx = (threadIdx.x - Wk + 1) / stride;
	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + small_tidx;
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((warpId == (WARP_COUNT - 1)) && (small_tidx >= 0) && (small_tidx < THIS_BLOCK_PROCESS_DATA_COUNT) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
	{
        for(int k = 0; k < (WARP_COUNT - 1); ++k)
            total_sum += prov[k*THIS_BLOCK_PROCESS_DATA_COUNT + small_tidx];

        output[index] = total_sum;
	}
}

// BATCH CONVOLUTION TEXTURE
template <typename T, int Hk>
__global__ void batch_convolution_tex_kernel(const T* image, const int Hi, const int Wi, const int D, image_wrapper<T> filters, const int num_filters, 
	const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int batch_index = blockIdx.z / num_filters;
	const int filter_index = blockIdx.z % num_filters;
	filters += filter_index*D*Hk*Wk;
	image += batch_index * D * Hi * Wi;

	T data[Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();

	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	for(int d = threadIdx.y; d < D; d += blockDim.y)
	{
		int index = d * Hi * Wi + tidy * Wi + tidx;

		for(int s = 0; s < Hk; ++s)
		{
			int tidy_ = tidy + s;
			if(tidx < Wi && tidy_ < Hi)
				data[s] = image[index];
			else
				data[s] = phast::get_zero<T>::get();
			index += Wi;
		}

		T sum = phast::get_zero<T>::get();

		for(int m = 0; m < Wk; ++m)
		{
			if(m > 0)
				sum = __shfl_up_sync(0xffffffff, sum, 1);

//			#pragma unroll
			for(int n = 0; n < Hk; ++n)
			{
				sum = data[n] * filters[d*Hk*Wk + n*Wk + m] + sum;
			}
		}

		total_sum += sum;
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_tex_kernel_bigHk_filter(const T* image, const int Hi, const int Wi, const int D, image_wrapper<T> filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int batch_index = blockIdx.z / num_filters;
	const int filter_index = blockIdx.z % num_filters;
	filters += filter_index*D*Hk*Wk;
	image += batch_index * D * Hi * Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();
	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk, tidy += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		for(int d = threadIdx.y; d < D; d += blockDim.y)
		{
			int index = d * Hi * Wi + tidy * Wi + tidx;
	
			for(int s = 0; s < h_range; ++s)
			{
				int tidy_ = tidy + s;
				if(tidx < Wi && tidy_ < Hi)
					data[s] = image[index];
				else
					data[s] = phast::get_zero<T>::get();
				index += Wi;
			}
	
			T sum = phast::get_zero<T>::get();
	
			for(int m = 0; m < Wk; ++m)
			{
				if(m > 0)
					sum = __shfl_up_sync(0xffffffff, sum, 1);
	
//				#pragma unroll
				for(int n = 0; n < h_range; ++n)
				{
					sum = data[n] * filters[d*Hk*Wk + (h1 + n)*Wk + m] + sum;
				}
			}
	
			total_sum += sum;
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}


template <typename T, int RED_Hk>
__global__ void batch_convolution_tex_kernel_big_filter(const T* image, const int Hi, const int Wi, const int D, image_wrapper<T> filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int BLOCK_PROCESS_DATA_COUNT = (blockDim.x - Wk + 1) / stride;
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;

	int THIS_BLOCK_PROCESS_DATA_COUNT;
	if((blockIdx.x + 1)*BLOCK_PROCESS_DATA_COUNT <= Wo)
		THIS_BLOCK_PROCESS_DATA_COUNT = BLOCK_PROCESS_DATA_COUNT;
	else
		THIS_BLOCK_PROCESS_DATA_COUNT = Wo - blockIdx.x*BLOCK_PROCESS_DATA_COUNT;

	const int batch_index = blockIdx.z / num_filters;
	const int filter_index = blockIdx.z % num_filters;
	filters += filter_index*D*Hk*Wk;
	image += batch_index*D*Hi*Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	int tidx, tidy;

	T* shared = SharedMemory<T>();
	T* prov = shared + blockDim.y;
	for(int i = threadIdx.x; i < (WARP_COUNT - 1)*THIS_BLOCK_PROCESS_DATA_COUNT; i += blockDim.x)
		prov[i] = phast::get_zero<T>::get();
	__syncthreads();

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + threadIdx.x;
		tidy = stride*blockIdx.y + h1;

		for(int d = threadIdx.y; d < D; d += blockDim.y)
		{
			int index = d * Hi * Wi + tidy * Wi + tidx;

			for(int s = 0; s < h_range; ++s)
			{
				int tidy_ = tidy + s;
				if(tidx < Wi && tidy_ < Hi)
					data[s] = image[index];
				else
					data[s] = phast::get_zero<T>::get();
				index += Wi;
			}

			T sum = phast::get_zero<T>::get();

			int p = THIS_BLOCK_PROCESS_DATA_COUNT;
			if(warpId == 0)
			{
				for(int m = 0; m < _PHAST_CUDA_WARP_SIZE; ++m)
				{
					if(m > 0)
						sum = __shfl_up_sync(0xffffffff, sum, 1);

					for(int n = 0; n < h_range; ++n)
						sum = data[n] * filters[d*Hk*Wk + (h1 + n)*Wk + m] + sum;

		            if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
		                prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
				}
			}
			else if(warpId != (WARP_COUNT -1))
			{
				const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
				for(int m = off; m < _PHAST_CUDA_WARP_SIZE; ++m)
				{
					if(m > off)
						sum = __shfl_up_sync(0xffffffff, sum, 1);

					if(laneId == 0)
						sum = phast::get_zero<T>::get();

					for(int n = 0; n < h_range; ++n)
						sum = data[n] * filters[d*Hk*Wk + (h1 + n)*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;

		            if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
		                prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
				}
			}
			else
			{
				const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
				for(int m = off; m < (Wk % _PHAST_CUDA_WARP_SIZE); ++m)
				{
					if(m > off)
						sum = __shfl_up_sync(0xffffffff, sum, 1);
					
					if(laneId == 0)
						sum = phast::get_zero<T>::get();

					for(int n = 0; n < h_range; ++n)
						sum = data[n] * filters[d*Hk*Wk + (h1 + n)*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;
				}
			}

			total_sum += sum;
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	int small_tidx = (threadIdx.x - Wk + 1) / stride;
	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + small_tidx;
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
    if((warpId == (WARP_COUNT - 1)) && (small_tidx >= 0) && (small_tidx < THIS_BLOCK_PROCESS_DATA_COUNT) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
	{
        for(int k = 0; k < (WARP_COUNT - 1); ++k)
            total_sum += prov[k*THIS_BLOCK_PROCESS_DATA_COUNT + small_tidx];

        output[index] = total_sum;
	}
}

template <typename T, int Hk>
__global__ void batch_convolution_tex_cm_kernel(const T* image, const int Hi, const int Wi, const int D, image_wrapper<T> filters, const int num_filters, 
	const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int filter_index = blockIdx.z / batch_size;
	const int batch_index = blockIdx.z % batch_size;
	filters += filter_index*Hk*Wk;
	image += batch_index*Hi*Wi;

	T data[Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();

	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	for(int d = threadIdx.y; d < D; d += blockDim.y)
	{
		int index = d * batch_size * Hi * Wi + tidy * Wi + tidx;

		for(int s = 0; s < Hk; ++s)
		{
			int tidy_ = tidy + s;
			if(tidx < Wi && tidy_ < Hi)
				data[s] = image[index];
			else
				data[s] = phast::get_zero<T>::get();
			index += Wi;
		}

		T sum = phast::get_zero<T>::get();

		for(int m = 0; m < Wk; ++m)
		{
			if(m > 0)
				sum = __shfl_up_sync(0xffffffff, sum, 1);

//			#pragma unroll
			for(int n = 0; n < Hk; ++n)
			{
				sum = data[n] * filters[d*num_filters*Hk*Wk + n*Wk + m] + sum;
			}
		}

		total_sum += sum;
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_tex_cm_kernel_bigHk_filter(const T* image, const int Hi, const int Wi, const int D, image_wrapper<T> filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int WARP_PROCESS_DATA_COUNT = 1 + (_PHAST_CUDA_WARP_SIZE - Wk) / stride;
	const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

	const int filter_index = blockIdx.z / batch_size;
	const int batch_index = blockIdx.z % batch_size;
	filters += filter_index*Hk*Wk;
	image += batch_index*Hi*Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	T* shared = SharedMemory<T>();

	int tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + WARP_PROCESS_DATA_COUNT*stride*warpId + laneId;
	int tidy = stride*blockIdx.y;

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk, tidy += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		for(int d = threadIdx.y; d < D; d += blockDim.y)
		{
			int index = d * batch_size * Hi * Wi + tidy * Wi + tidx;

			for(int s = 0; s < h_range; ++s)
			{
				int tidy_ = tidy + s;
				if(tidx < Wi && tidy_ < Hi)
					data[s] = image[index];
				else
					data[s] = phast::get_zero<T>::get();
				index += Wi;
			}

			T sum = phast::get_zero<T>::get();

			for(int m = 0; m < Wk; ++m)
			{
				if(m > 0)
					sum = __shfl_up_sync(0xffffffff, sum, 1);

//				#pragma unroll
				for(int n = 0; n < h_range; ++n)
				{
					sum = data[n] * filters[d*num_filters*Hk*Wk + (h1 + n)*Wk + m] + sum;
				}
			}

			total_sum += sum;
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + ((laneId - Wk + 1) / stride);
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((laneId >= Wk-1) && ((laneId - Wk + 1) % stride == 0) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
		output[index] = total_sum;
}

template <typename T, int RED_Hk>
__global__ void batch_convolution_tex_cm_kernel_big_filter(const T* image, const int Hi, const int Wi, const int D, image_wrapper<T> filters, const int num_filters, 
	const int Hk, const int Wk, T* output, const int Ho, const int Wo, const int stride, const T* bias_ptr, const int batch_size)
{
	const int laneId = threadIdx.x & (_PHAST_CUDA_WARP_SIZE - 1);
	const int warpId = threadIdx.x / _PHAST_CUDA_WARP_SIZE;
	const int BLOCK_PROCESS_DATA_COUNT = (blockDim.x - Wk + 1) / stride;
	const int WARP_COUNT = blockDim.x / _PHAST_CUDA_WARP_SIZE;

    int THIS_BLOCK_PROCESS_DATA_COUNT;
    if((blockIdx.x + 1)*BLOCK_PROCESS_DATA_COUNT <= Wo)
        THIS_BLOCK_PROCESS_DATA_COUNT = BLOCK_PROCESS_DATA_COUNT;
    else
        THIS_BLOCK_PROCESS_DATA_COUNT = Wo - blockIdx.x*BLOCK_PROCESS_DATA_COUNT;

	const int filter_index = blockIdx.z /batch_size;
	const int batch_index = blockIdx.z % batch_size;
	filters += filter_index*Hk*Wk;
	image += batch_index*Hi*Wi;

	T data[RED_Hk]; // register cache
	T total_sum = phast::get_zero<T>::get(); // sum

	int tidx, tidy;

	T* shared = SharedMemory<T>();
	T* prov = shared + blockDim.y;
	for(int i = threadIdx.x; i < (WARP_COUNT - 1)*BLOCK_PROCESS_DATA_COUNT; i += blockDim.x)
		prov[i] = phast::get_zero<T>::get();
	__syncthreads();

	for(int h1 = 0; h1 < Hk; h1 += RED_Hk)
	{
		int h_range = (h1 + RED_Hk < Hk) ? (RED_Hk) : (Hk-h1);

		tidx = BLOCK_PROCESS_DATA_COUNT*stride*blockIdx.x + threadIdx.x;
		tidy = stride*blockIdx.y + h1;

		for(int d = threadIdx.y; d < D; d += blockDim.y)
		{
			int index = d * batch_size * Hi * Wi + tidy * Wi + tidx;

			for(int s = 0; s < h_range; ++s)
			{
				int tidy_ = tidy + s;
				if(tidx < Wi && tidy_ < Hi)
					data[s] = image[index];
				else
					data[s] = phast::get_zero<T>::get();
				index += Wi;
			}

			T sum = phast::get_zero<T>::get();

			int p = THIS_BLOCK_PROCESS_DATA_COUNT;
            if(warpId == 0)
            {
                for(int m = 0; m < _PHAST_CUDA_WARP_SIZE; ++m)
                {
                    if(m > 0)
                        sum = __shfl_up_sync(0xffffffff, sum, 1);

                    for(int n = 0; n < h_range; ++n)
                        sum = data[n] * filters[d*num_filters*Hk*Wk + (h1 + n)*Wk + m] + sum;

                    if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
                        prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
                }
            }
            else if(warpId != (WARP_COUNT -1))
            {
                const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
                for(int m = off; m < _PHAST_CUDA_WARP_SIZE; ++m)
                {
                    if(m > off)
                        sum = __shfl_up_sync(0xffffffff, sum, 1);

                    if(laneId == 0)
                        sum = phast::get_zero<T>::get();

                    for(int n = 0; n < h_range; ++n)
                        sum = data[n] * filters[d*num_filters*Hk*Wk + (h1 + n)*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;

                    if((laneId == _PHAST_CUDA_WARP_SIZE-1) && (m >= _PHAST_CUDA_WARP_SIZE - THIS_BLOCK_PROCESS_DATA_COUNT))
                        prov[warpId*THIS_BLOCK_PROCESS_DATA_COUNT + (--p)] += sum;
                }
            }
            else
            {
                const int off = -THIS_BLOCK_PROCESS_DATA_COUNT + 1;
                for(int m = off; m < (Wk % _PHAST_CUDA_WARP_SIZE); ++m)
                {
                    if(m > off)
                        sum = __shfl_up_sync(0xffffffff, sum, 1);

                    if(laneId == 0)
                        sum = phast::get_zero<T>::get();

                    for(int n = 0; n < h_range; ++n)
                        sum = data[n] * filters[d*num_filters*Hk*Wk + (h1 + n)*Wk + warpId*_PHAST_CUDA_WARP_SIZE + m] + sum;
                }
            }

			total_sum += sum;
		}
	}

	if(threadIdx.x == 0)
		shared[threadIdx.y] = total_sum;
	__syncthreads();

	total_sum = phast::get_zero<T>::get();
	for(int k = 0; k < blockDim.y; ++k)
		total_sum += shared[k];

	if(bias_ptr != nullptr)
		total_sum += bias_ptr[filter_index];

	int small_tidx = (threadIdx.x - Wk + 1) / stride;
	tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + small_tidx;
	tidy = blockIdx.y;
	int index = blockIdx.z * Ho * Wo + tidy * Wo + tidx;
	if((warpId == (WARP_COUNT - 1)) && (small_tidx >= 0) && (small_tidx < THIS_BLOCK_PROCESS_DATA_COUNT) && (tidx < Wo) && (tidy < Ho) && (threadIdx.y == 0))
	{
        for(int k = 0; k < (WARP_COUNT - 1); ++k)
            total_sum += prov[k*THIS_BLOCK_PROCESS_DATA_COUNT + small_tidx];

        output[index] = total_sum;
	}
}

#define _INVOKE_BATCH_CONV(Hk) \
	batch_convolution_kernel<T, Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters.get_dev(), \
		num_filters, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CONV \
	switch(Hk) \
	{ \
        case 1: {_INVOKE_BATCH_CONV(1)} break; \
        case 2: {_INVOKE_BATCH_CONV(2)} break; \
        case 3: {_INVOKE_BATCH_CONV(3)} break; \
        case 4: {_INVOKE_BATCH_CONV(4)} break; \
        case 5: {_INVOKE_BATCH_CONV(5)} break; \
        case 6: {_INVOKE_BATCH_CONV(6)} break; \
        case 7: {_INVOKE_BATCH_CONV(7)} break; \
        case 8: {_INVOKE_BATCH_CONV(8)} break; \
        case 9: {_INVOKE_BATCH_CONV(9)} break; \
        case 10: {_INVOKE_BATCH_CONV(10)} break; \
        case 11: {_INVOKE_BATCH_CONV(11)} break; \
        case 12: {_INVOKE_BATCH_CONV(12)} break; \
        case 13: {_INVOKE_BATCH_CONV(13)} break; \
        case 14: {_INVOKE_BATCH_CONV(14)} break; \
        case 15: {_INVOKE_BATCH_CONV(15)} break; \
        case 16: {_INVOKE_BATCH_CONV(16)} break; \
        case 17: {_INVOKE_BATCH_CONV(17)} break; \
        case 18: {_INVOKE_BATCH_CONV(18)} break; \
        case 19: {_INVOKE_BATCH_CONV(19)} break; \
        case 20: {_INVOKE_BATCH_CONV(20)} break; \
        case 21: {_INVOKE_BATCH_CONV(21)} break; \
        case 22: {_INVOKE_BATCH_CONV(22)} break; \
        case 23: {_INVOKE_BATCH_CONV(23)} break; \
        case 24: {_INVOKE_BATCH_CONV(24)} break; \
        case 25: {_INVOKE_BATCH_CONV(25)} break; \
        case 26: {_INVOKE_BATCH_CONV(26)} break; \
        case 27: {_INVOKE_BATCH_CONV(27)} break; \
        case 28: {_INVOKE_BATCH_CONV(28)} break; \
        case 29: {_INVOKE_BATCH_CONV(29)} break; \
        case 30: {_INVOKE_BATCH_CONV(30)} break; \
        case 31: {_INVOKE_BATCH_CONV(31)} break; \
        case 32: {_INVOKE_BATCH_CONV(32)} break; \
	} /**/
#define _INVOKE_BATCH_CONV_BIGHK_FILTER(REDUCED_Hk) \
	batch_convolution_kernel_bigHk_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters.get_dev(), \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CONV_BIGHK_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CONV_BIGHK_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CONV_BIGHK_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CONV_BIGHK_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CONV_BIGHK_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CONV_BIGHK_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CONV_BIGHK_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CONV_BIGHK_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CONV_BIGHK_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CONV_BIGHK_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CONV_BIGHK_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CONV_BIGHK_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CONV_BIGHK_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CONV_BIGHK_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CONV_BIGHK_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CONV_BIGHK_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CONV_BIGHK_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CONV_BIGHK_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CONV_BIGHK_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CONV_BIGHK_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CONV_BIGHK_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CONV_BIGHK_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CONV_BIGHK_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CONV_BIGHK_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CONV_BIGHK_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CONV_BIGHK_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CONV_BIGHK_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CONV_BIGHK_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CONV_BIGHK_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CONV_BIGHK_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CONV_BIGHK_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CONV_BIGHK_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CONV_BIGHK_FILTER(32)} break; \
	}    /**/ 
#define _INVOKE_BATCH_CONV_BIG_FILTER(REDUCED_Hk) \
	batch_convolution_kernel_big_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters.get_dev(), \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CONV_BIG_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CONV_BIG_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CONV_BIG_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CONV_BIG_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CONV_BIG_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CONV_BIG_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CONV_BIG_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CONV_BIG_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CONV_BIG_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CONV_BIG_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CONV_BIG_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CONV_BIG_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CONV_BIG_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CONV_BIG_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CONV_BIG_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CONV_BIG_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CONV_BIG_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CONV_BIG_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CONV_BIG_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CONV_BIG_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CONV_BIG_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CONV_BIG_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CONV_BIG_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CONV_BIG_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CONV_BIG_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CONV_BIG_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CONV_BIG_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CONV_BIG_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CONV_BIG_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CONV_BIG_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CONV_BIG_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CONV_BIG_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CONV_BIG_FILTER(32)} break; \
	}    /**/

#define _INVOKE_BATCH_CONV_TEX(Hk) \
	batch_convolution_tex_kernel<T, Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters_wrap.image, \
		num_filters, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CONV_TEX \
	switch(Hk) \
	{ \
        case 1: {_INVOKE_BATCH_CONV_TEX(1)} break; \
        case 2: {_INVOKE_BATCH_CONV_TEX(2)} break; \
        case 3: {_INVOKE_BATCH_CONV_TEX(3)} break; \
        case 4: {_INVOKE_BATCH_CONV_TEX(4)} break; \
        case 5: {_INVOKE_BATCH_CONV_TEX(5)} break; \
        case 6: {_INVOKE_BATCH_CONV_TEX(6)} break; \
        case 7: {_INVOKE_BATCH_CONV_TEX(7)} break; \
        case 8: {_INVOKE_BATCH_CONV_TEX(8)} break; \
        case 9: {_INVOKE_BATCH_CONV_TEX(9)} break; \
        case 10: {_INVOKE_BATCH_CONV_TEX(10)} break; \
        case 11: {_INVOKE_BATCH_CONV_TEX(11)} break; \
        case 12: {_INVOKE_BATCH_CONV_TEX(12)} break; \
        case 13: {_INVOKE_BATCH_CONV_TEX(13)} break; \
        case 14: {_INVOKE_BATCH_CONV_TEX(14)} break; \
        case 15: {_INVOKE_BATCH_CONV_TEX(15)} break; \
        case 16: {_INVOKE_BATCH_CONV_TEX(16)} break; \
        case 17: {_INVOKE_BATCH_CONV_TEX(17)} break; \
        case 18: {_INVOKE_BATCH_CONV_TEX(18)} break; \
        case 19: {_INVOKE_BATCH_CONV_TEX(19)} break; \
        case 20: {_INVOKE_BATCH_CONV_TEX(20)} break; \
        case 21: {_INVOKE_BATCH_CONV_TEX(21)} break; \
        case 22: {_INVOKE_BATCH_CONV_TEX(22)} break; \
        case 23: {_INVOKE_BATCH_CONV_TEX(23)} break; \
        case 24: {_INVOKE_BATCH_CONV_TEX(24)} break; \
        case 25: {_INVOKE_BATCH_CONV_TEX(25)} break; \
        case 26: {_INVOKE_BATCH_CONV_TEX(26)} break; \
        case 27: {_INVOKE_BATCH_CONV_TEX(27)} break; \
        case 28: {_INVOKE_BATCH_CONV_TEX(28)} break; \
        case 29: {_INVOKE_BATCH_CONV_TEX(29)} break; \
        case 30: {_INVOKE_BATCH_CONV_TEX(30)} break; \
        case 31: {_INVOKE_BATCH_CONV_TEX(31)} break; \
        case 32: {_INVOKE_BATCH_CONV_TEX(32)} break; \
	} /**/
#define _INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(REDUCED_Hk) \
	batch_convolution_tex_kernel_bigHk_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters_wrap.image, \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CONV_TEX_BIGHK_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CONV_TEX_BIGHK_FILTER(32)} break; \
	}    /**/ 
#define _INVOKE_BATCH_CONV_TEX_BIG_FILTER(REDUCED_Hk) \
	batch_convolution_tex_kernel_big_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters_wrap.image, \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CONV_TEX_BIG_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CONV_TEX_BIG_FILTER(32)} break; \
	}    /**/ 

#define _INVOKE_BATCH_CM_CONV(Hk) \
	batch_convolution_cm_kernel<T, Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters.get_dev(), \
		num_filters, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CM_CONV \
	switch(Hk) \
	{ \
        case 1: {_INVOKE_BATCH_CM_CONV(1)} break; \
        case 2: {_INVOKE_BATCH_CM_CONV(2)} break; \
        case 3: {_INVOKE_BATCH_CM_CONV(3)} break; \
        case 4: {_INVOKE_BATCH_CM_CONV(4)} break; \
        case 5: {_INVOKE_BATCH_CM_CONV(5)} break; \
        case 6: {_INVOKE_BATCH_CM_CONV(6)} break; \
        case 7: {_INVOKE_BATCH_CM_CONV(7)} break; \
        case 8: {_INVOKE_BATCH_CM_CONV(8)} break; \
        case 9: {_INVOKE_BATCH_CM_CONV(9)} break; \
        case 10: {_INVOKE_BATCH_CM_CONV(10)} break; \
        case 11: {_INVOKE_BATCH_CM_CONV(11)} break; \
        case 12: {_INVOKE_BATCH_CM_CONV(12)} break; \
        case 13: {_INVOKE_BATCH_CM_CONV(13)} break; \
        case 14: {_INVOKE_BATCH_CM_CONV(14)} break; \
        case 15: {_INVOKE_BATCH_CM_CONV(15)} break; \
        case 16: {_INVOKE_BATCH_CM_CONV(16)} break; \
        case 17: {_INVOKE_BATCH_CM_CONV(17)} break; \
        case 18: {_INVOKE_BATCH_CM_CONV(18)} break; \
        case 19: {_INVOKE_BATCH_CM_CONV(19)} break; \
        case 20: {_INVOKE_BATCH_CM_CONV(20)} break; \
        case 21: {_INVOKE_BATCH_CM_CONV(21)} break; \
        case 22: {_INVOKE_BATCH_CM_CONV(22)} break; \
        case 23: {_INVOKE_BATCH_CM_CONV(23)} break; \
        case 24: {_INVOKE_BATCH_CM_CONV(24)} break; \
        case 25: {_INVOKE_BATCH_CM_CONV(25)} break; \
        case 26: {_INVOKE_BATCH_CM_CONV(26)} break; \
        case 27: {_INVOKE_BATCH_CM_CONV(27)} break; \
        case 28: {_INVOKE_BATCH_CM_CONV(28)} break; \
        case 29: {_INVOKE_BATCH_CM_CONV(29)} break; \
        case 30: {_INVOKE_BATCH_CM_CONV(30)} break; \
        case 31: {_INVOKE_BATCH_CM_CONV(31)} break; \
        case 32: {_INVOKE_BATCH_CM_CONV(32)} break; \
	} /**/
#define _INVOKE_BATCH_CM_CONV_BIGHK_FILTER(REDUCED_Hk) \
	batch_convolution_cm_kernel_bigHk_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters.get_dev(), \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CM_CONV_BIGHK_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CM_CONV_BIGHK_FILTER(32)} break; \
	}    /**/ 
#define _INVOKE_BATCH_CM_CONV_BIG_FILTER(REDUCED_Hk) \
	batch_convolution_cm_kernel_big_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters.get_dev(), \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CM_CONV_BIG_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CM_CONV_BIG_FILTER(32)} break; \
	}    /**/ 

#define _INVOKE_BATCH_CM_CONV_TEX(Hk) \
	batch_convolution_tex_cm_kernel<T, Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters_wrap.image, \
		num_filters, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CM_CONV_TEX \
	switch(Hk) \
	{ \
        case 1: {_INVOKE_BATCH_CM_CONV_TEX(1)} break; \
        case 2: {_INVOKE_BATCH_CM_CONV_TEX(2)} break; \
        case 3: {_INVOKE_BATCH_CM_CONV_TEX(3)} break; \
        case 4: {_INVOKE_BATCH_CM_CONV_TEX(4)} break; \
        case 5: {_INVOKE_BATCH_CM_CONV_TEX(5)} break; \
        case 6: {_INVOKE_BATCH_CM_CONV_TEX(6)} break; \
        case 7: {_INVOKE_BATCH_CM_CONV_TEX(7)} break; \
        case 8: {_INVOKE_BATCH_CM_CONV_TEX(8)} break; \
        case 9: {_INVOKE_BATCH_CM_CONV_TEX(9)} break; \
        case 10: {_INVOKE_BATCH_CM_CONV_TEX(10)} break; \
        case 11: {_INVOKE_BATCH_CM_CONV_TEX(11)} break; \
        case 12: {_INVOKE_BATCH_CM_CONV_TEX(12)} break; \
        case 13: {_INVOKE_BATCH_CM_CONV_TEX(13)} break; \
        case 14: {_INVOKE_BATCH_CM_CONV_TEX(14)} break; \
        case 15: {_INVOKE_BATCH_CM_CONV_TEX(15)} break; \
        case 16: {_INVOKE_BATCH_CM_CONV_TEX(16)} break; \
        case 17: {_INVOKE_BATCH_CM_CONV_TEX(17)} break; \
        case 18: {_INVOKE_BATCH_CM_CONV_TEX(18)} break; \
        case 19: {_INVOKE_BATCH_CM_CONV_TEX(19)} break; \
        case 20: {_INVOKE_BATCH_CM_CONV_TEX(20)} break; \
        case 21: {_INVOKE_BATCH_CM_CONV_TEX(21)} break; \
        case 22: {_INVOKE_BATCH_CM_CONV_TEX(22)} break; \
        case 23: {_INVOKE_BATCH_CM_CONV_TEX(23)} break; \
        case 24: {_INVOKE_BATCH_CM_CONV_TEX(24)} break; \
        case 25: {_INVOKE_BATCH_CM_CONV_TEX(25)} break; \
        case 26: {_INVOKE_BATCH_CM_CONV_TEX(26)} break; \
        case 27: {_INVOKE_BATCH_CM_CONV_TEX(27)} break; \
        case 28: {_INVOKE_BATCH_CM_CONV_TEX(28)} break; \
        case 29: {_INVOKE_BATCH_CM_CONV_TEX(29)} break; \
        case 30: {_INVOKE_BATCH_CM_CONV_TEX(30)} break; \
        case 31: {_INVOKE_BATCH_CM_CONV_TEX(31)} break; \
        case 32: {_INVOKE_BATCH_CM_CONV_TEX(32)} break; \
	} /**/
#define _INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(REDUCED_Hk) \
	batch_convolution_tex_cm_kernel_bigHk_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters_wrap.image, \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CM_CONV_TEX_BIGHK_FILTER(32)} break; \
	}    /**/ 
#define _INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(REDUCED_Hk) \
	batch_convolution_tex_cm_kernel_big_filter<T, REDUCED_Hk><<<grid_size, block_size, smem_size>>>(images.get_dev(), Hi, Wi, D, filters_wrap.image, \
		num_filters, Hk, Wk, output.get_dev(), Ho, Wo, stride, bias_ptr, batch_size);
#define INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER \
	switch(hk) /* Reduced Hk*/ \
	{ \
        case 1: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(1)} break; \
        case 2: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(2)} break; \
        case 3: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(3)} break; \
        case 4: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(4)} break; \
        case 5: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(5)} break; \
        case 6: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(6)} break; \
        case 7: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(7)} break; \
        case 8: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(8)} break; \
        case 9: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(9)} break; \
        case 10: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(10)} break; \
        case 11: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(11)} break; \
        case 12: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(12)} break; \
        case 13: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(13)} break; \
        case 14: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(14)} break; \
        case 15: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(15)} break; \
        case 16: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(16)} break; \
        case 17: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(17)} break; \
        case 18: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(18)} break; \
        case 19: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(19)} break; \
        case 20: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(20)} break; \
        case 21: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(21)} break; \
        case 22: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(22)} break; \
        case 23: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(23)} break; \
        case 24: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(24)} break; \
        case 25: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(25)} break; \
        case 26: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(26)} break; \
        case 27: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(27)} break; \
        case 28: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(28)} break; \
        case 29: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(29)} break; \
        case 30: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(30)} break; \
        case 31: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(31)} break; \
        case 32: {_INVOKE_BATCH_CM_CONV_TEX_BIG_FILTER(32)} break; \
	}    /**/ 


template <typename T>
void batch_convolution(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, int batch_size, 
	const phast::vector<T>* bias, phast::cube<T>& output)
{
	cudaError_t ret;

	const int D = filters.size_i() / num_filters;
	const int Hi = images.size_j();
	const int Wi = images.size_k();
	const int Hk = filters.size_j();
	const int Wk = filters.size_k();
	const int Ho = output.size_j();
	const int Wo = output.size_k();

	const T* bias_ptr = (bias != nullptr) ? (bias->get_dev()) : nullptr;

	int block_size_x, block_size_y, smem_size, hk, wk;
	adjust_block_size<T>(block_size_x, block_size_y, smem_size, Hk, Wk, D, stride, hk, wk);

	if((Hk == hk) && (Wk == wk)) // small filter
	{
	    const int WARP_COUNT = block_size_x / _PHAST_CUDA_WARP_SIZE; // warp-count
		const int WARP_PROCESS_DATA_COUNT = 1 + _PHAST_MAX((_PHAST_CUDA_WARP_SIZE - Wk) / stride, 0);
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

		const int grid_size_x = iDivUp(Wo, BLOCK_PROCESS_DATA_COUNT);
		const int grid_size_y = Ho;
		const int grid_size_z = num_filters * batch_size;

		const dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
		const dim3 block_size(block_size_x, block_size_y, 1);
		INVOKE_BATCH_CONV;
		CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
	}
	else if(Wk == wk)
	{
	    const int WARP_COUNT = block_size_x / _PHAST_CUDA_WARP_SIZE; // warp-count
		const int WARP_PROCESS_DATA_COUNT = 1 + _PHAST_MAX((_PHAST_CUDA_WARP_SIZE - Wk) / stride, 0);
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

		const int grid_size_x = iDivUp(Wo, BLOCK_PROCESS_DATA_COUNT);
		const int grid_size_y = Ho;
		const int grid_size_z = num_filters * batch_size;

		const dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
		const dim3 block_size(block_size_x, block_size_y, 1);
		INVOKE_BATCH_CONV_BIGHK_FILTER;
		CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
	}
	else
	{
		const int BLOCK_PROCESS_DATA_COUNT = _PHAST_MAX(1, (block_size_x - Wk) / stride);

		const int grid_size_x = iDivUp(Wo, BLOCK_PROCESS_DATA_COUNT);
		const int grid_size_y = Ho;
		const int grid_size_z = num_filters * batch_size;

		const dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
		const dim3 block_size(block_size_x, block_size_y, 1);
		INVOKE_BATCH_CONV_BIG_FILTER;
		CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
	}
}

template <typename T>
void batch_convolution_channel_major(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, int batch_size, 
	const phast::vector<T>* bias, phast::cube<T>& output)
{
	cudaError_t ret;

	const int D = filters.size_i() / num_filters;
	const int Hi = images.size_j();
	const int Wi = images.size_k();
	const int Hk = filters.size_j();
	const int Wk = filters.size_k();
	const int Ho = output.size_j();
	const int Wo = output.size_k();

	const T* bias_ptr = (bias != nullptr) ? (bias->get_dev()) : nullptr;

	int block_size_x, block_size_y, smem_size, hk, wk;
	adjust_block_size<T>(block_size_x, block_size_y, smem_size, Hk, Wk, D, stride, hk, wk);

	if((Hk == hk) && (Wk == wk)) // small filter
	{
    	const int WARP_COUNT = block_size_x / _PHAST_CUDA_WARP_SIZE; // warp-count
		const int WARP_PROCESS_DATA_COUNT = 1 + _PHAST_MAX((_PHAST_CUDA_WARP_SIZE - Wk) / stride, 0);
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

		const int grid_size_x = iDivUp(Wo, BLOCK_PROCESS_DATA_COUNT);
		const int grid_size_y = Ho;
		const int grid_size_z = num_filters * batch_size;

		const dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
		const dim3 block_size(block_size_x, block_size_y, 1);
		INVOKE_BATCH_CM_CONV;
		CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
	}
	else if(Wk == wk)
	{
    	const int WARP_COUNT = block_size_x / _PHAST_CUDA_WARP_SIZE; // warp-count
		const int WARP_PROCESS_DATA_COUNT = 1 + _PHAST_MAX((_PHAST_CUDA_WARP_SIZE - Wk) / stride, 0);
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT * WARP_COUNT;

		const int grid_size_x = iDivUp(Wo, BLOCK_PROCESS_DATA_COUNT);
		const int grid_size_y = Ho;
		const int grid_size_z = num_filters * batch_size;

		const dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
		const dim3 block_size(block_size_x, block_size_y, 1);
		INVOKE_BATCH_CM_CONV_BIGHK_FILTER;
		CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
	}
	else // big filter
	{
		const int BLOCK_PROCESS_DATA_COUNT = _PHAST_MAX(1, (block_size_x - Wk) / stride);

		const int grid_size_x = iDivUp(Wo, BLOCK_PROCESS_DATA_COUNT);
		const int grid_size_y = Ho;
		const int grid_size_z = num_filters * batch_size;

		const dim3 grid_size(grid_size_x, grid_size_y, grid_size_z);
		const dim3 block_size(block_size_x, block_size_y, 1);
		INVOKE_BATCH_CM_CONV_BIG_FILTER;
		CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
	}
}

// ROTATE AND PAD
template <typename T>
inline void adjust_block_size_rotate_and_pad(int& block_size, int& smem_size, int sizei, int& grid_size)
{
    block_size = phast::custom::cuda::get_major_block_size();
	if(block_size & (_PHAST_CUDA_WARP_SIZE - 1) != 0)
		block_size -= (block_size & (_PHAST_CUDA_WARP_SIZE - 1));
	if(block_size < _PHAST_CUDA_WARP_SIZE)
		block_size = _PHAST_CUDA_WARP_SIZE;
	if(block_size > _PHAST_CUDA_MAX_THREAD_PER_BLOCK)
		block_size = _PHAST_CUDA_MAX_THREAD_PER_BLOCK;

	smem_size = block_size * sizeof(T);
	grid_size = sizei;
}

template <typename T>
__global__ void rotate_and_pad_kernel(const T* src, T* dst, const int I, const int H, const int W, const int ph, const int pw,
	const int HH, const int WW)
{
	const int N_round = (H*W + blockDim.x - 1) / blockDim.x;
	T* shared = SharedMemory<T>();
	src += blockIdx.x * H * W;
	dst += blockIdx.x * HH * WW + ph * WW + pw;

	int tix = threadIdx.x;
	for(int r = 0; r < N_round; ++r, tix += blockDim.x)
	{
		const int B = (((H*W) % blockDim.x == 0) || (r != N_round - 1)) ? (blockDim.x) : ((H*W) % blockDim.x);
		if(tix < H*W)
			shared[threadIdx.x] = src[tix];
		__syncthreads();

		int j = tix / W;
		int k = tix % W;
		if(tix < H*W)
			dst[j*WW + k] = shared[B -1 -threadIdx.x];
	}
}

template <typename T>
void rotate_and_pad(const phast::cube<T>& src, const int ph, const int pw, const int HH, const int WW, phast::cube<T>& dst)
{
	cudaError_t ret;

	int block_size, smem_size, grid_size;
	adjust_block_size_rotate_and_pad<T>(block_size, smem_size, src.size_i(), grid_size);

	cudaMemset(dst.get_dev(), 0, dst.size_i()*HH*WW*sizeof(T));
	rotate_and_pad_kernel<<<grid_size, block_size, smem_size>>>(src.get_dev(), dst.get_dev(), src.size_i(), src.size_j(), src.size_k(), ph, pw, HH, WW);
	CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
}

template <typename T>
__global__ void pad_kernel(const T* src, T* dst, const int I, const int H, const int W, const int ph, const int pw,
	const int HH, const int WW)
{
	src += blockIdx.x * H * W;
	dst += blockIdx.x * HH * WW + ph * WW + pw;

	for(int tix = threadIdx.x; tix < (H*W); tix += blockDim.x)
	{
		int j = tix / W;
		int k = tix % W;
		if(tix < H*W)
			dst[j*WW + k] = src[tix];
	}
}

template <typename T>
void pad(const phast::cube<T>& src, const int ph, const int pw, const int HH, const int WW, phast::cube<T>& dst)
{
	cudaError_t ret;

	int block_size, smem_size, grid_size;
	adjust_block_size_rotate_and_pad<T>(block_size, smem_size, src.size_i(), grid_size);

	cudaMemset(dst.get_dev(), 0, dst.size_i()*HH*WW*sizeof(T));
	pad_kernel<<<grid_size, block_size>>>(src.get_dev(), dst.get_dev(), src.size_i(), src.size_j(), src.size_k(), ph, pw, HH, WW);
	CUDA_DEVICE_SYNCH(ret, __FILE__, __LINE__);
}

}
}
}

#endif /* __AI_CUDA_H */

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

#ifndef __AI_MULTI_H
#define __AI_MULTI_H

#include <phast.h>

namespace phast
{
namespace ai
{
namespace multi
{

template <typename T, unsigned int policy = phast::get_default_policy()>
struct transposer : phast::functor::func_mat<T, policy>
{
    _PHAST_METHOD transposer(const phast::cube<T>& src, const int num_batches, const int num_filters)
    {
        src_.link(src);
        num_batches_ = num_batches;
        num_filters_ = num_filters;
    }
    _PHAST_METHOD void operator()(phast::functor::matrix<T>& mat)
    {
        const int I = this->get_index() / num_batches_;
        const int J = this->get_index() % num_batches_;

        auto src_mat = *(src_.cbegin_i() + J*num_filters_ + I);
        this->copy(src_mat.cbegin_ij(), src_mat.cend_ij(), mat.begin_ij());
    }

    phast::functor::cube<T> src_;
    int num_batches_;
    int num_filters_;
};

template <typename T>
void batch_conv_thread(const phast::cube<T>* images, const phast::cube<T>* filters, int num_filters, int stride, 
	int batch_size, const phast::vector<T>* bias, phast::cube<T>* output, int thread_id, int thread_dim)
{
	phast::multi::set_affinity(thread_id);
	int last = std::min<int>((thread_id + 1) * thread_dim, num_filters * batch_size);

	const int D = filters->size_i() / num_filters;
	const int Hi = images->size_j();
	const int Wi = images->size_k();
	const int Hk = filters->size_j();
	const int Wk = filters->size_k();
	const int Ho = output->size_j();
	const int Wo = output->size_k();

	const T* images_ptr = images->get_dev();
	const T* bias_ptr = bias ? bias->get_dev() : nullptr;
	T* output_ptr = output->get_dev();
	
	for(int m = thread_id * thread_dim; m < last; ++m)
	{
		const int batch_index = m / num_filters;
		const int filter_index = m % num_filters;

		const T* filter_ptr = filters->get_dev() + filter_index*D*Hk*Wk;
		if(bias_ptr != nullptr)
		{
			// apply bias
			for(int h = 0; h < Ho; ++h)
				for(int w = 0; w < Wo; ++w)
					output_ptr[m*Ho*Wo + h*Wo + w] = bias_ptr[filter_index];
		}

		for(int d = 0; d < D; ++d)
			for(int h = 0; h < Ho; ++h)
				for(int y = 0; y < Hk; ++y)
					for(int w = 0; w < Wo; ++w)
						for(int x = 0; x < Wk; ++x)
							output_ptr[m*Ho*Wo + h*Wo + w] += images_ptr[(batch_index*D + d)*Hi*Wi + (h*stride+y)*Wi + (w*stride+x)] * filter_ptr[d*Hk*Wk + y*Wk + x];
	}
}

template <typename T>
void batch_convolution(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, int batch_size, 
	const phast::vector<T>* bias, phast::cube<T>& output)
{
	int N_thread, thread_dim;

	// 1st possible implementation: parallelize over the filters - it makes sense if there are many of them
	{
		phast::multi::set_sizes<void>(&N_thread, &thread_dim, num_filters * batch_size);
		phast::multi::thread_group threads;
		for(int j = 0; j < N_thread; ++j)
			threads.add_thread(phast::multi::fast_join_thread(batch_conv_thread<T>, &images, &filters, num_filters, stride, batch_size, 
				bias, &output, j, thread_dim));
		threads.join_all();	
	}

	// other possible solutions...
}

template <typename T>
void batch_convolution_channel_major(const phast::cube<T>& images, const phast::cube<T>& filters, int num_filters, int stride, int batch_size, 
	const phast::vector<T>* bias, phast::cube<T>& output)
{
    phast::cube<T> images_transp(images.size_i(), images.size_j(), images.size_k());
    phast::cube<T> filters_transp(filters.size_i(), filters.size_j(), filters.size_k());
    phast::cube<T> output_transp(output.size_i(), output.size_j(), output.size_k());

	const int D = filters.size_i() / num_filters;
    for_each(images_transp.begin_i(), images_transp.end_i(), transposer<T>(images, D, batch_size));
    for_each(filters_transp.begin_i(), filters_transp.end_i(), transposer<T>(filters, D, num_filters));

	batch_convolution(images_transp, filters_transp, num_filters, stride, batch_size, bias, output_transp);

	for_each(output.begin_i(), output.end_i(), transposer<T>(output_transp, batch_size, num_filters));
}

template <typename T>
void rotate_and_pad_thread(const phast::cube<T>* src, const int ph, const int pw, phast::cube<T>* dst, const int HH, const int WW,
	int thread_id, int thread_dim)
{
	phast::multi::set_affinity(thread_id);
	int last = std::min<int>((thread_id + 1) * thread_dim, src->size_i());

	const int H = src->size_j();
	const int W = src->size_k();

	const T* src_ptr = src->get_dev();
	T* dst_ptr = dst->get_dev() + (ph + H-1)*WW + pw + W-1;

	for(int i = thread_id * thread_dim; i< last; ++i)
	{
		for(int j = 0; j < H; ++j)
		{
			for(int k = 0; k < W; ++k)
			{
				dst_ptr[i*HH*WW + -j*WW + -k] = src_ptr[i*H*W + j*W + k];
			}
		}
	}
}

template <typename T>
void rotate_and_pad(const phast::cube<T>& src, const int ph, const int pw, const int HH, const int WW, phast::cube<T>& dst)
{
	int N_thread, thread_dim;
	phast::multi::set_sizes<void>(&N_thread, &thread_dim, src.size_i());
	phast::multi::thread_group threads;

	memset(dst.get_dev(), 0, dst.size_i()*HH*WW*sizeof(T));
	for(int j = 0; j < N_thread; ++j)
		threads.add_thread(phast::multi::fast_join_thread(rotate_and_pad_thread<T>, &src, ph, pw, &dst, HH, WW, j, thread_dim));
	threads.join_all();	
}

template <typename T>
void pad_thread(const phast::cube<T>* src, const int ph, const int pw, phast::cube<T>* dst, const int HH, const int WW,
	int thread_id, int thread_dim)
{
	phast::multi::set_affinity(thread_id);
	int last = std::min<int>((thread_id + 1) * thread_dim, src->size_i());

	const int H = src->size_j();
	const int W = src->size_k();

	const T* src_ptr = src->get_dev();
	T* dst_ptr = dst->get_dev() + ph*WW + pw;

	for(int i = thread_id * thread_dim; i< last; ++i)
	{
		for(int j = 0; j < H; ++j)
		{
			for(int k = 0; k < W; ++k)
			{
				dst_ptr[i*HH*WW + j*WW + k] = src_ptr[i*H*W + j*W + k];
			}
		}
	}
}

template <typename T>
void pad(const phast::cube<T>& src, const int ph, const int pw, const int HH, const int WW, phast::cube<T>& dst)
{
	int N_thread, thread_dim;
	phast::multi::set_sizes<void>(&N_thread, &thread_dim, src.size_i());
	phast::multi::thread_group threads;

	memset(dst.get_dev(), 0, dst.size_i()*HH*WW*sizeof(T));
	for(int j = 0; j < N_thread; ++j)
		threads.add_thread(phast::multi::fast_join_thread(pad_thread<T>, &src, ph, pw, &dst, HH, WW, j, thread_dim));
	threads.join_all();	
}

}
}
}

#endif /* __AI_MULTI_H */

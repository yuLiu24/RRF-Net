#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bi_directional_loss_layer.hpp"

using std::max;

using namespace std;
using namespace cv;

namespace caffe {

//int myrandom (int i) { return caffe_rng_rand()%i;}


template <typename Dtype>
void BiDirectionalLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  img_diff_.ReshapeLike(*bottom[0]);
  text_diff_.ReshapeLike(*bottom[2]);
  dis_.Reshape(bottom[0]->num(), bottom[2]->num(), 1, 1);
  mask_.Reshape(bottom[0]->num(), bottom[2]->num(), 1, 1);
  mask2_.Reshape(bottom[2]->num(), bottom[0]->num(), 1, 1);
}


template <typename Dtype>
void BiDirectionalLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom)
{

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	float alpha1 = rank_param.alpha1();
        float alpha2 = rank_param.alpha2();
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	int hard_num = neg_num * hard_ratio;
	int rand_num = neg_num * rand_ratio;

	const Dtype* img_data = bottom[0]->cpu_data();
	const Dtype* img_label = bottom[1]->cpu_data();
        const Dtype* text_data = bottom[2]->cpu_data();
        const Dtype* text_label = bottom[3]->cpu_data();

	int img_count = bottom[0]->count();
	int img_num = bottom[0]->num();
	int img_dim = bottom[0]->count() / bottom[0]->num();

	int text_count = bottom[2]->count();
	int text_num = bottom[2]->num();
	int text_dim = bottom[2]->count() / bottom[2]->num();

	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data(); // rows: img; cols: text
        Dtype* mask_data2 = mask2_.mutable_cpu_data(); // rows: text; cols: img

        int num = img_num;

	for(int i = 0; i < num * num; i ++)
	{
		dis_data[i] = 0;
		mask_data[i] = 0;
                mask_data2[i] = 0;
	}

	// calculate distance
	for(int i = 0; i < num; i ++)
	{
		for(int j = 0; j < num; j ++)
		{
			const Dtype* fea1 = img_data + i * img_dim;
			const Dtype* fea2 = text_data + j * text_dim;
			Dtype ts = 0;
			for(int k = 0; k < img_dim; k ++)
			{
			  ts += (fea1[k] * fea2[k]) ;
			}
			dis_data[i * text_num + j] = -ts;
		}
	}

	//select samples

	vector<pair<float, int> >i2t_negpairs;
        vector<pair<float, int> >t2i_negpairs;

	for(int i = 0; i < num; i ++)
	{
		i2t_negpairs.clear();
                t2i_negpairs.clear();
             
		for(int j = 0; j < num; j ++)
		{
                        // image-to-text negative pairs
			if(img_label[i] != text_label[j])
                        {
			   Dtype tloss1 = alpha1 * max(Dtype(0), dis_data[i * num + i] - dis_data[i * num + j] + Dtype(margin));
			   if(tloss1 > 0) 
			      i2t_negpairs.push_back(make_pair(dis_data[i * num + j], j));
                        }
                        // text-to-image negative pairs
 			if(text_label[i] != img_label[j])
                        {             
                           Dtype tloss2 = alpha2 * max(Dtype(0), dis_data[i * num + i] - dis_data[j * num + i] + Dtype(margin));
                           if(tloss2 > 0) 
                              t2i_negpairs.push_back(make_pair(dis_data[j * num + i], j));
                        }
		}
                // if valid negpairs in batch size are less than the expected neg_num 
		if(i2t_negpairs.size() <= neg_num)
		{
			for(int j = 0; j < i2t_negpairs.size(); j ++)
			{
				int id = i2t_negpairs[j].second;
				mask_data[i * num + id] = 1;
			}
		}
                else {
                    // else valid negpairs in batch size are more than the expected neg_num
		    sort(i2t_negpairs.begin(), i2t_negpairs.end());
		    for(int j = 0; j < neg_num; j ++)
		    {
			mask_data[i * num + i2t_negpairs[j].second] = 1;
		    }
                }

		if(t2i_negpairs.size() <= neg_num)
		{
			for(int j = 0; j < t2i_negpairs.size(); j ++)
			{
				int id = t2i_negpairs[j].second;
				mask_data2[i * num + id] = 1;
			}
		}
                else {
                    sort(t2i_negpairs.begin(), t2i_negpairs.end());
		    for(int j = 0; j < neg_num; j ++)
		    {
                        mask_data2[i * num + t2i_negpairs[j].second] = 1;
		    }
                }
	}

}

template <typename Dtype>
void BiDirectionalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	//const Dtype* img_data = bottom[0]->cpu_data();
	//const Dtype* img_label = bottom[1]->cpu_data();
        //const Dtype* text_data = bottom[2]->cpu_data();
        //const Dtype* text_label = bottom[3]->cpu_data();

	//int img_count = bottom[0]->count();
	int num = bottom[0]->num();
	//int img_dim = bottom[0]->count() / bottom[0]->num();

	//int text_count = bottom[2]->count();
	//int text_num = bottom[2]->num();
	//int text_dim = bottom[2]->count() / bottom[2]->num();

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();      // 4
	float alpha1 = rank_param.alpha1();  // 5
        float alpha2 = rank_param.alpha2();  // 5
	//float hard_ratio = rank_param.hard_ratio();
	//float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();
        Dtype* mask_data2 = mask2_.mutable_cpu_data();

	set_mask(bottom);
	Dtype loss = 0;

        int cnt = neg_num * num;

	for(int i = 0; i < num; i ++)
	{
		for(int j = 0; j < num; j ++)
		{
                    Dtype tloss1 = 0;
                    Dtype tloss2 = 0;
		    if(mask_data[i * num + j] == 1) 
                    {
		       tloss1 = alpha1 * max(Dtype(0), dis_data[i * num + i] - dis_data[i * num + j] + Dtype(margin));
                    }
		    if(mask_data2[i * num + j] == 1) 
                    {
		       tloss2 = alpha2 * max(Dtype(0), dis_data[i * num + i] - dis_data[j * num + i] + Dtype(margin));
                    }
                     
		    loss += tloss1 + tloss2;
		}
	}

	loss = loss / cnt;
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BiDirectionalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* img_data = bottom[0]->cpu_data();
	//const Dtype* img_label = bottom[1]->cpu_data();
        const Dtype* text_data = bottom[2]->cpu_data();
        //const Dtype* text_label = bottom[3]->cpu_data();

	int img_count = bottom[0]->count();
	int img_num = bottom[0]->num();
	int img_dim = bottom[0]->count() / bottom[0]->num();

	int text_count = bottom[2]->count();
	int text_num = bottom[2]->num();
	int text_dim = bottom[2]->count() / bottom[2]->num();
 
        int num = img_num;

        Dtype* img_diff = bottom[0]->mutable_cpu_diff();
        Dtype* text_diff = bottom[2]->mutable_cpu_diff();

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	float alpha1 = rank_param.alpha1();
        float alpha2 = rank_param.alpha2();
	//float hard_ratio = rank_param.hard_ratio();
	//float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();
        Dtype* mask_data2 = mask2_.mutable_cpu_data();

	for(int i = 0; i < img_count; i ++ )
        {
		img_diff[i] = 0;
                text_diff[i] = 0;
        }

        int cnt = neg_num * img_num;

	for(int i = 0; i < num; i ++)
	{
                const Dtype* img_pos = img_data + i * img_dim;
	        Dtype* img_pos_diff = img_diff + i * img_dim;
                const Dtype* text_pos = text_data + i * text_dim;
                Dtype* text_pos_diff = text_diff + i * text_dim;

		for(int j = 0; j < num; j ++)
		{
                    // image-to-text
		    if(mask_data[i * num + j] == 1) 
                    {			 
                          const Dtype* text_neg = text_data + j * text_dim;
			  Dtype* text_neg_diff = text_diff + j * text_dim;

                        Dtype tloss1 = max(Dtype(0), dis_data[i * img_num + i] - dis_data[i * img_num + j] + Dtype(margin));
			if(tloss1 > 0)
                        {
			  for(int k = 0; k < img_dim; k ++)
			  {
			      img_pos_diff[k] += alpha1 * (text_neg[k] - text_pos[k]); 
			      text_pos_diff[k] += -alpha1 * img_pos[k]; 
			      text_neg_diff[k] += alpha1 * img_pos[k];
			  }
                        }
                    }

                    // text-to-image
		    if(mask_data2[i * num + j] == 1) 
                    {
                          const Dtype* img_neg = img_data + j * img_dim;
			  Dtype* img_neg_diff = img_diff + j * img_dim;

			  Dtype tloss2 = max(Dtype(0), dis_data[i * img_num + i] - dis_data[j * img_num + i] + Dtype(margin));
			if(tloss2 > 0)
                        {
			  for(int k = 0; k < text_dim; k ++)
			  {
			      text_pos_diff[k] += alpha2 * (img_neg[k] - img_pos[k]); 
			      img_pos_diff[k] += -alpha2 * text_pos[k]; 
			      img_neg_diff[k] += alpha2 * text_pos[k];
			  }
                        }
                    }
		}
	}

	for (int i = 0; i < img_count; i ++)
	{
		img_diff[i] = img_diff[i] / cnt;
                text_diff[i] = text_diff[i] / cnt;
	}

}

#ifdef CPU_ONLY
STUB_GPU(BiDirectionalLossLayer);
#endif

INSTANTIATE_CLASS(BiDirectionalLossLayer);
REGISTER_LAYER_CLASS(BiDirectionalLoss);

}  // namespace caffe

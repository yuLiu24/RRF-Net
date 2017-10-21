#ifndef CAFFE_BI_RANK_LOSS_LAYER_HPP_
#define CAFFE_BI_RANK_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
//#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class BiRankLossLayer : public LossLayer<Dtype> {
 public:
  explicit BiRankLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), img_diff_(), text_diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline const char* type() const { return "BiRankLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void set_mask(const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> mask_;
  Blob<Dtype> mask2_;
  Blob<Dtype> dis_;

  Blob<Dtype> img_diff_;
  Blob<Dtype> text_diff_;

};

}
#endif  // CAFFE_BI_RANK_LOSS_LAYER_HPP_

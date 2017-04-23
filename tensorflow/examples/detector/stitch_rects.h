#ifndef STITCH_RECTS_HPP
#define STITCH_RECTS_HPP

#include <math.h>
#include <stdlib.h>
#include <vector>

#include "hungarian.h"
#include "tensorflow/core/framework/tensor.h"

//#define MIN(a, b) (((a) < (b)) ? (a) : (b))
//#define MAX(a, b) (((a) > (b)) ? (a) : (b))

namespace tensorbox {

class Rect {
 public:
  int cx_;
  int cy_;
  int width_;
  int height_;
  int class_id_;
  float depth_;
  float confidence_;
  float true_confidence_;

  explicit Rect(int cx, int cy, int width, int height, float confidence,
                int class_id, int depth) {
    cx_ = cx;
    cy_ = cy;
    width_ = width;
    height_ = height;
    class_id_ = class_id;
    depth_ = depth;
    confidence_ = confidence;
    true_confidence_ = confidence;
  }

  Rect(const Rect &other) {
    cx_ = other.cx_;
    cy_ = other.cy_;
    width_ = other.width_;
    height_ = other.height_;
    class_id_ = other.class_id_;
    depth_ = other.depth_;
    confidence_ = other.confidence_;
    true_confidence_ = other.true_confidence_;
  }

  bool overlaps(const Rect &other, float tau) const {
    if (fabs(cx_ - other.cx_) > (width_ + other.width_) / 1.5) {
      return false;
    } else if (fabs(cy_ - other.cy_) > (height_ + other.height_) / 2.0) {
      return false;
    } else {
      return iou(other) > tau;
    }
  }

  int distance(const Rect &other) const {
    return (fabs(cx_ - other.cx_) + fabs(cy_ - other.cy_) +
            fabs(width_ - other.width_) + fabs(height_ - other.height_));
  }

  float intersection(const Rect &other) const {
    int left = MAX(cx_ - width_ / 2., other.cx_ - other.width_ / 2.);
    int right = MIN(cx_ + width_ / 2., other.cx_ + other.width_ / 2.);
    int width = MAX(right - left, 0);

    int top = MAX(cy_ - height_ / 2., other.cy_ - other.height_ / 2.);
    int bottom = MIN(cy_ + height_ / 2., other.cy_ + other.height_ / 2.);
    int height = MAX(bottom - top, 0);
    return width * height;
  }

  int area() const { return height_ * width_; }

  float union_area(const Rect &other) const {
    return this->area() + other.area() - this->intersection(other);
  }

  float iou(const Rect &other) const {
    return this->intersection(other) / this->union_area(other);
  }

  bool operator==(const Rect &other) const {
    return (cx_ == other.cx_ && cy_ == other.cy_ && width_ == other.width_ &&
            class_id_ == other.class_id_ && height_ == other.height_ &&
            confidence_ == other.confidence_);
  }
};

void filter_rects(
    const std::vector<std::vector<std::vector<Rect> > > &all_rects,
    std::vector<Rect> *stitched_rects, float threshold, float max_threshold,
    float tau, float conf_alpha);

void stitch_rects(const Eigen::Tensor<float, 4, Eigen::RowMajor> &boxes,
                  const Eigen::Tensor<float, 4, Eigen::RowMajor> &confidences,
                  float tau, int region_size, std::vector<Rect> *out);
}  // namespace Tensorbox

#endif  // STITCH_RECTS_HPP

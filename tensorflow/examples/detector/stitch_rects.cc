#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "hungarian.h"
#include "stitch_rects.h"

using std::vector;

namespace tensorbox {

const std::vector<std::pair<float, float> > thresholds = {
    {.80, 1.0}, {.70, 0.9}, {.60, 0.8}, {.50, 0.7}, {.40, 0.6},  {.30, 0.5},
    {.20, 0.4}, {.10, 0.3}, {.05, 0.2}, {.02, 0.1}, {.005, 0.4}, {.001, 0.01}};

void filter_rects(const vector<vector<vector<Rect> > > &all_rects,
                  vector<Rect> *stitched_rects, float threshold,
                  float max_threshold, float tau, float conf_alpha) {
  const vector<Rect> &accepted_rects = *stitched_rects;
  for (int i = 0; i < (int)all_rects.size(); ++i) {
    for (int j = 0; j < (int)all_rects[0].size(); ++j) {
      vector<Rect> current_rects;
      for (int k = 0; k < (int)all_rects[i][j].size(); ++k) {
        if (all_rects[i][j][k].confidence_ * conf_alpha > threshold) {
          Rect r = Rect(all_rects[i][j][k]);
          r.confidence_ *= conf_alpha;
          r.true_confidence_ *= conf_alpha;
          current_rects.push_back(r);
        }
      }

      vector<Rect> relevant_rects;
      for (int k = 0; k < (int)accepted_rects.size(); ++k) {
        for (int l = 0; l < (int)current_rects.size(); ++l) {
          if (accepted_rects[k].overlaps(current_rects[l], tau)) {
            relevant_rects.push_back(Rect(accepted_rects[k]));
            break;
          }
        }
      }

      if (relevant_rects.size() == 0 || current_rects.size() == 0) {
        for (int k = 0; k < (int)current_rects.size(); ++k) {
          stitched_rects->push_back(Rect(current_rects[k]));
        }
        continue;
      }

      int num_pred = MAX(current_rects.size(), relevant_rects.size());

      int int_cost[num_pred * num_pred];
      for (int k = 0; k < num_pred * num_pred; ++k) {
        int_cost[k] = 0;
      }
      for (int k = 0; k < (int)current_rects.size(); ++k) {
        for (int l = 0; l < (int)relevant_rects.size(); ++l) {
          int idx = k * num_pred + l;
          int cost = 10000;
          if (current_rects[k].overlaps(relevant_rects[l], tau)) {
            cost -= 1000;
          }
          cost += (int)(current_rects[k].distance(relevant_rects[l]) / 10.);
          int_cost[idx] = cost;
        }
      }

      std::vector<int> assignment;

      hungarian_problem_t p;
      int **m = array_to_matrix(int_cost, num_pred, num_pred);
      hungarian_init(&p, m, num_pred, num_pred, HUNGARIAN_MODE_MINIMIZE_COST);
      hungarian_solve(&p);
      for (int i = 0; i < num_pred; ++i) {
        for (int j = 0; j < num_pred; ++j) {
          if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
            assignment.push_back(j);
          }
        }
      }
      assert((int)assignment.size() == num_pred);
      hungarian_free(&p);

      for (int i = 0; i < num_pred; ++i) {
        free(m[i]);
      }
      free(m);

      vector<int> bad;
      for (int k = 0; k < (int)assignment.size(); ++k) {
        Rect &c = current_rects[k];
        Rect &a = relevant_rects[assignment[k]];
        if (c.confidence_ > max_threshold) {
          bad.push_back(k);
          continue;
        }
        if (c.overlaps(a, tau)) {
          if (c.confidence_ > a.confidence_ && c.iou(a) > 0.7) {
            c.true_confidence_ = a.confidence_;
            stitched_rects->erase(
                std::find(stitched_rects->begin(), stitched_rects->end(), a));
          } else {
            bad.push_back(k);
          }
        }
      }

      for (int k = 0; k < (int)current_rects.size(); ++k) {
        bool bad_contains_k = false;
        for (int l = 0; l < (int)bad.size(); ++l) {
          if (k == bad[l]) {
            bad_contains_k = true;
            break;
          }
        }
        if (!bad_contains_k) {
          stitched_rects->push_back(Rect(current_rects[k]));
        }
      }
    }
  }
}

int get_class_id(const Eigen::Tensor<float, 4, Eigen::RowMajor> &confidences,
                 int i, int j, int k) {
  int class_id = 1;
  for (int c = 2; c < confidences.dimension(3); ++c) {
    if (confidences(i, j, k, c) > confidences(i, j, k, class_id)) {
      class_id = c;
    }
  }

  return class_id;
}

void stitch_rects(const Eigen::Tensor<float, 4, Eigen::RowMajor> &boxes,
                  const Eigen::Tensor<float, 4, Eigen::RowMajor> &confidences,
                  float tau, int region_size, std::vector<Rect> *out) {
  // convert tensor to vector
  std::vector<std::vector<std::vector<Rect> > > all_rects;
  all_rects.resize(boxes.dimension(0));
  for (int i = 0; i < boxes.dimension(0); ++i) {
    all_rects[i].resize(boxes.dimension(1));
    for (int j = 0; j < boxes.dimension(1); ++j) {
      for (int k = 0; k < boxes.dimension(2); ++k) {
        const int class_id = get_class_id(confidences, i, j, k);
        const float depth = (boxes.dimension(3) == 4) ? -1 : boxes(i, j, k, 4);
        all_rects[i][j].emplace_back(Rect(static_cast<int>(boxes(i, j, k, 0)) +
                                              region_size / 2 + region_size * j,
                                          static_cast<int>(boxes(i, j, k, 1)) +
                                              region_size / 2 + region_size * i,
                                          boxes(i, j, k, 2), boxes(i, j, k, 3),
                                          confidences(i, j, k, class_id),
                                          class_id, depth));
      }
    }
  }

  // filter rects
  const std::vector<std::pair<float, float> > t_conf_alphas = {{tau, 1.0}};
  for (const auto &conf : t_conf_alphas) {
    for (const auto &threshold : thresholds) {
      if (threshold.first * conf.second > 0.0001) {
        filter_rects(all_rects, out, threshold.first * conf.second,
                     threshold.second * conf.second, conf.first, conf.second);
      }
    }
  }
}

}  // namespace tensorbox

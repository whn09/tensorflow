#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "opencv2/opencv.hpp"
#include "stitch_rects.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace cv;

Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  tensorflow::SessionOptions options;
  // options.config.set_allow_soft_placement(true);
  session->reset(tensorflow::NewSession(options));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

void printUsage() {
  LOG(INFO) << "--------------------------------------";
  LOG(INFO) << "  j: previous rectangle";
  LOG(INFO) << "  k: next rectangle";
  LOG(INFO) << "  f: toggle final result";
  LOG(INFO) << "  q: quit";
  LOG(INFO) << "  ?: this message";
  LOG(INFO) << "--------------------------------------";
}

class MultiClassRect {
 public:
  int cx_;
  int cy_;
  int width_;
  int height_;
  std::vector<float> confidences_;
};

const int region_size = 32;
const int rnn_len = 5;
const int box_region = 64;

void getAllRects(const Eigen::Tensor<float, 4, Eigen::RowMajor> &boxes,
                 const Eigen::Tensor<float, 4, Eigen::RowMajor> &confidences,
                 std::vector<MultiClassRect> *all_rects) {
  for (int i = 0; i < boxes.dimension(0); ++i) {
    for (int j = 0; j < boxes.dimension(1); ++j) {
      for (int k = 0; k < boxes.dimension(2); ++k) {
        MultiClassRect rect;
        rect.cx_ = static_cast<int>(boxes(i, j, k, 0)) + region_size / 2 +
                   region_size * j,
        rect.cy_ = static_cast<int>(boxes(i, j, k, 1)) + region_size / 2 +
                   region_size * i,
        rect.width_ = boxes(i, j, k, 2);
        rect.height_ = boxes(i, j, k, 3);
        for (int c = 0; c < confidences.dimension(3); ++c) {
          rect.confidences_.push_back(confidences(i, j, k, c));
        }

        all_rects->emplace_back(rect);
      }
    }
  }
}

void renderRect(const Mat &input, const std::vector<MultiClassRect> &all_rects,
                const std::vector<tensorbox::Rect> &stitched_rects, int idx,
                float conf, bool show_final) {
  int grid_width = input.cols / region_size;
  int grid_idx = idx / rnn_len;
  int grid_idx_w = grid_idx % grid_width;
  int grid_idx_h = grid_idx / grid_width;
  int x = (grid_idx_w + 0.5) * region_size;
  int y = (grid_idx_h + 0.5) * region_size;

  Mat render = input.clone();
  circle(render, Point(x, y), 5, Scalar(255, 0, 0), -1, CV_AA);
  rectangle(render, Rect(x - box_region / 2, y - box_region / 2, box_region,
                         box_region),
            Scalar(255, 0, 0), 1, CV_AA);

  const auto &rect = all_rects[idx];
  circle(render, Point(rect.cx_, rect.cy_), 5, Scalar(0, 255, 0), -1, CV_AA);
  rectangle(render,
            Rect(rect.cx_ - rect.width_ / 2, rect.cy_ - rect.height_ / 2,
                 rect.width_, rect.height_),
            Scalar(0, 255, 0), 2);

  if (show_final) {
    for (const auto &rect : stitched_rects) {
      if (rect.confidence_ > conf) {
        rectangle(render,
                  Rect(rect.cx_ - rect.width_ / 2, rect.cy_ - rect.height_ / 2,
                       rect.width_, rect.height_),
                  Scalar(0, 0, 255), 1);
      }
    }
  }

  std::ostringstream oss;
  oss << idx + 1 << "/" << all_rects.size() << " (";
  oss << std::setprecision(3) << rect.confidences_[0];
  if (rect.confidences_[0] >= conf) {
    oss << "*";
  }
  for (int i = 1; i < (int)rect.confidences_.size(); ++i) {
    oss << ", " << std::setprecision(3) << rect.confidences_[i];
    if (rect.confidences_[i] >= conf) {
      oss << "*";
    }
  }
  oss << ")";
  putText(render, oss.str(), Point(20, render.rows - 40), FONT_HERSHEY_SIMPLEX,
          0.5, Scalar(0, 255, 0), 1, 8);

  imshow("Object detection", render);
}

int main(int argc, char *argv[]) {
  //-----------------------------------
  // parse args
  string graph_path =
      "/home/yanli/data/project/tensorbox/data/"
      "pedestrian_frozen_graph_2016_07_29_18.35.pb";
  int img_width = 640;
  int img_height = 384;
  int num_classes = 2;
  string scale = "0.5";
  string conf = "0.1";
  string inter = "linear";
  string img;
  bool show_final = true;

  string input_layer = "x_in";
  std::vector<string> output_layers = {"add", "Reshape_2"};

  const bool parse_result = tensorflow::Flags::Parse(
      &argc, argv, {Flag("graph", &graph_path,"graph path"), Flag("img_width", &img_width, "image width"),
                    Flag("img_height", &img_height, "image widht"), Flag("scale", &scale, "scale"),
                    Flag("conf", &conf, "confidence"), Flag("inter", &inter, "inter"),
                    Flag("num_classes", &num_classes, "number classes"), Flag("img", &img, "image")});
  if (!parse_result) {
    LOG(ERROR) << "Error parsing command-line flags.";
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1];
    return -1;
  }

  //-----------------------------------
  // load tensorflow model
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  LOG(INFO) << "Graph loaded: " << graph_path;

  //-----------------------------------
  // load image and run model
  const float conf_threshold = std::stof(conf);
  const float scale_param = std::stof(scale);
  int interpolation = (inter == "linear") ? INTER_LINEAR : INTER_CUBIC;

  Mat res;
  Mat src = imread(img, CV_LOAD_IMAGE_COLOR);
  resize(src, res, Size(0, 0), scale_param, scale_param, interpolation);
  Mat input = Mat::zeros(img_height, img_width, CV_8UC3);
  Rect roi =
      Rect(0, 0, std::min(res.cols, img_width), std::min(res.rows, img_height));
  res(roi).copyTo(input(roi));

  Tensor input_tensor(tensorflow::DT_FLOAT,
                      tensorflow::TensorShape({img_height, img_width, 3}));
  auto input_tensor_mapped = input_tensor.tensor<float, 3>();
  for (int y = 0; y < img_height; ++y) {
    for (int x = 0; x < img_width; ++x) {
      Vec3b pixel = input.at<Vec3b>(y, x);
      input_tensor_mapped(y, x, 0) = pixel.val[2];
      input_tensor_mapped(y, x, 1) = pixel.val[1];
      input_tensor_mapped(y, x, 2) = pixel.val[0];
    }
  }

  std::vector<Tensor> out_tensors;
  Status run_status = session->Run({{input_layer, input_tensor}}, output_layers,
                                   {}, &out_tensors);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  const int grid_width = img_width / region_size;
  const int grid_height = img_height / region_size;
  auto boxes =
      out_tensors[0].shaped<float, 4>({grid_height, grid_width, rnn_len, 4});
  auto confidences = out_tensors[1].shaped<float, 4>(
      {grid_height, grid_width, rnn_len, num_classes});

  std::vector<tensorbox::Rect> out;
  stitch_rects(boxes, confidences, 0.25, region_size, &out);

  std::vector<MultiClassRect> all_rects;
  getAllRects(boxes, confidences, &all_rects);

  session->Close();
  printUsage();

  //-----------------------------------
  // render result
  int idx = 0;
  for (; idx < (int)all_rects.size();) {
    renderRect(input, all_rects, out, idx, conf_threshold, show_final);

    char key = waitKey(0) & 0xFF;
    if (key == 'q') {
      break;
    } else if (key == 'j') {
      idx = max(idx - 1, 0);
    } else if (key == 'k') {
      idx = min((int)all_rects.size() - 1, idx + 1);
    } else if (key == 'f') {
      show_final = !show_final;
    } else if (key == '?') {
      printUsage();
    }
  }

  return 0;
}

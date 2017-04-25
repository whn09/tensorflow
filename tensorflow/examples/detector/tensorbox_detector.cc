#include <jni.h>
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

#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/replace.hpp"
#include "boost/filesystem.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"
#include "opencv2/opencv.hpp"
#include "stitch_rects.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace cv;

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::high_resolution_clock::time_point;

#define TENSORBOXDETECTOR_METHOD(METHOD_NAME) \
  Java_com_ucarinc_adas_android_TensorboxDetector_##METHOD_NAME  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
TENSORBOXDETECTOR_METHOD(test_image)(
    JNIEnv* env, jclass clazz, jint input, jint output);

#ifdef __cplusplus
}
#endif

JNIEXPORT void JNICALL
TENSORBOXDETECTOR_METHOD(test_image)(
    JNIEnv* env, jclass clazz, jint input, jint output) {
        output = input+1;
    }

// -------------------------------------------------------------------------
struct Args {
  bool show_img = true;
  bool dump_result = false;
  int img_width = 640;
  int img_height = 384;
  int num_classes = 2;
  int region_size = 32;
  int gpu = 0;
  int num_boxes = 5;
  int shape_dim = 4;
  string eval_list;
  string eval_classes;
  string video_path;
  string graph_path;
  string scale = "0.5";
  string conf = "0.1";
  string inter = "linear";
  string class_names = "car,ped";
  string input_layer = "x_in";
  string output_layers = "add,Reshape_2";
  string output;
  string training_config;
  string class_id_mapping = "1:1,2:2,3:3,4:20";
  // additional params
  int grid_width;
  int grid_height;
  int interpolation;
  float conf_threshold;
  float scale_param;
  std::vector<std::string> output_layer_tokens;
};

// -------------------------------------------------------------------------
Status load_graph(string graph_file_name, int gpu_id,
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
  std::ostringstream oss;
//  oss << "/gpu:" << gpu_id;
  oss << "/cpu:" << gpu_id;
  tensorflow::graph::SetDefaultDevice(oss.str(), &graph_def);
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// -------------------------------------------------------------------------
inline int run_detection(const Args &args, const float scale,
                         const Mat &src_img,
                         const std::unique_ptr<tensorflow::Session> &session,
                         Tensor &input_tensor, Mat &input, TimePoint &timer3,
                         TimePoint &timer4, std::vector<tensorbox::Rect> &out) {
  Mat res;
  resize(src_img, res, Size(0, 0), scale, scale, args.interpolation);
  Rect roi =
      Rect(0, 0, min(res.cols, args.img_width), min(res.rows, args.img_height));
  res(roi).copyTo(input(roi));

  auto input_tensor_mapped = input_tensor.tensor<float, 3>();
  for (int y = 0; y < args.img_height; ++y) {
    for (int x = 0; x < args.img_width; ++x) {
      Vec3b pixel = input.at<Vec3b>(y, x);
      input_tensor_mapped(y, x, 0) = pixel.val[2];
      input_tensor_mapped(y, x, 1) = pixel.val[1];
      input_tensor_mapped(y, x, 2) = pixel.val[0];
    }
  }

  timer3 = Clock::now();

  std::vector<Tensor> out_tensors;
  Status run_status = session->Run({{args.input_layer, input_tensor}},
                                   args.output_layer_tokens, {}, &out_tensors);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  auto boxes = out_tensors[0].shaped<float, 4>(
      {args.grid_height, args.grid_width, args.num_boxes, args.shape_dim});
  auto confidences = out_tensors[1].shaped<float, 4>(
      {args.grid_height, args.grid_width, args.num_boxes, args.num_classes});

  timer4 = Clock::now();

  stitch_rects(boxes, confidences, 0.25, args.region_size, &out);
  return 0;
}

// -------------------------------------------------------------------------
int test_video(const Args &args,
               const std::unique_ptr<tensorflow::Session> &session) {
  // open video
  VideoCapture inputVideo(args.video_path);
  if (!inputVideo.isOpened()) {
    LOG(ERROR) << "Could not open the input video: " << args.video_path;
    return -1;
  }

  // Get Codec Type- Int form
  int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));
  char EXT[] = {(char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8),
                (char)((ex & 0XFF0000) >> 16), (char)((ex & 0XFF000000) >> 24),
                0};

  // Acquire input size
  Size S = Size((int)inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
                (int)inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

  const int video_frame_num = inputVideo.get(CV_CAP_PROP_FRAME_COUNT);
  LOG(INFO) << "Input frame resolution: Width=" << S.width
            << "  Height=" << S.height << " of " << video_frame_num
            << " frames";
  LOG(INFO) << "Input codec type: " << EXT;

  Tensor input_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({args.img_height, args.img_width, 3}));

  // set up output video
  VideoWriter outputVideo;
  if (args.dump_result) {
    const string out_path =
        args.output.empty() ? "/tmp/result.mp4" : args.output;
    //change ex to CV_FOURCC('F','L','V','1')
//    outputVideo.open(out_path, CV_FOURCC('F','L','V','1'), inputVideo.get(CV_CAP_PROP_FPS),
    outputVideo.open(out_path, CV_FOURCC('A','V','C','1'), inputVideo.get(CV_CAP_PROP_FPS),
                     Size(args.img_width, args.img_height), true);
    if (!outputVideo.isOpened()) {
      LOG(ERROR) << "Can't open video " << out_path;
    }
  }

  // -------------------------------------
  // run detection
  std::vector<std::string> class_tokens;
  boost::split(class_tokens, args.class_names, boost::is_any_of(","));

  const std::vector<Scalar> object_colors = {
      Scalar(0, 255, 0),   Scalar(255, 0, 0),   Scalar(255, 255, 0),
      Scalar(0, 0, 128),   Scalar(0, 165, 255), Scalar(255, 0, 255),
      Scalar(130, 0, 75),  Scalar(255, 255, 0), Scalar(220, 20, 60),
      Scalar(138, 43, 226)};

  std::chrono::duration<double, std::milli> time_io(0);
  std::chrono::duration<double, std::milli> time_imgproc(0);
  std::chrono::duration<double, std::milli> time_nn(0);
  std::chrono::duration<double, std::milli> time_stitching(0);
  std::chrono::duration<double, std::milli> time_rendering(0);

  auto t1 = Clock::now();
  bool pause = false;
  int key_wait_duration = 1;
  int accumulated_frames = 0;
  // namedWindow("Object detection");  //, WINDOW_NORMAL);

  Mat src_img;
  for (int vid_idx = 0;;) {
    auto timer1 = Clock::now();

    inputVideo >> src_img;
    vid_idx++;
    accumulated_frames++;
    if (src_img.empty()) {
      break;
    }

    auto timer2 = Clock::now();

    TimePoint timer3;
    TimePoint timer4;
    std::vector<tensorbox::Rect> out;
    Mat input = Mat::zeros(args.img_height, args.img_width, CV_8UC3);
    run_detection(args, args.scale_param, src_img, session, input_tensor, input,
                  timer3, timer4, out);

    auto timer5 = Clock::now();

    // show result
    for (const auto &rect : out) {
      if (rect.confidence_ > args.conf_threshold) {
        auto color = object_colors[(rect.class_id_ - 1) % object_colors.size()];
        string title = (rect.class_id_ <= (int)class_tokens.size())
                           ? class_tokens[rect.class_id_ - 1]
                           : "unknown";
        std::ostringstream oss;
        if (args.shape_dim == 4) {
          oss << title << "(" << std::setprecision(3) << rect.confidence_
              << ")";
        } else {
          oss << title << "(" << std::setprecision(3) << rect.depth_ << "m)";
        }
        rectangle(input, cv::Rect(rect.cx_ - rect.width_ / 2,
                                  rect.cy_ - rect.height_ / 2, rect.width_,
                                  rect.height_),
                  color, 1);
        rectangle(input,
                  cv::Rect(rect.cx_ - rect.width_ / 2,
                           rect.cy_ - rect.height_ / 2 - 20, rect.width_, 20),
                  color, CV_FILLED);
        putText(input, oss.str(), Point(rect.cx_ - rect.width_ / 2 + 3,
                                        rect.cy_ - rect.height_ / 2 - 5),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 2, CV_AA);
        putText(input, oss.str(), Point(rect.cx_ - rect.width_ / 2 + 3,
                                        rect.cy_ - rect.height_ / 2 - 5),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1, CV_AA);
      }
    }

    auto t2 = Clock::now();
    using std::chrono::duration_cast;
    auto duration = duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    if (!args.show_img) {
      std::cout << '\r' << vid_idx << "/" << video_frame_num << ", "
                << 1.0e9 * accumulated_frames / duration << "fps" << std::flush;
      if (args.dump_result && outputVideo.isOpened()) {
        outputVideo << input;
      }
      continue;
    }

    std::ostringstream oss;
    oss << vid_idx << "/" << video_frame_num;
    if (!pause) {
      oss << ", " << std::setprecision(4)
          << 1.0e9 * accumulated_frames / duration << " fps";
    }
    putText(input, oss.str(), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5,
            Scalar(0, 0, 255), 1, CV_AA);

    //imshow("Object detection", input);

    auto timer6 = Clock::now();

    char key = waitKey(key_wait_duration) & 0xFF;
    if (key == 'q') {
      break;
    } else if (key == 'p') {
      pause = !pause;
      key_wait_duration = pause ? 0 : 1;
      if (!pause) {
        // reset all timers
        time_io = std::chrono::milliseconds(0);
        time_imgproc = std::chrono::milliseconds(0);
        time_nn = std::chrono::milliseconds(0);
        time_stitching = std::chrono::milliseconds(0);
        time_rendering = std::chrono::milliseconds(0);

        accumulated_frames = 0;
        t1 = Clock::now();
      }
    } else if (pause && key == 'j') {
      vid_idx = inputVideo.get(CV_CAP_PROP_POS_FRAMES);
      inputVideo.set(CV_CAP_PROP_POS_FRAMES, max(0, vid_idx - 2));
    } else if (pause && key == 'g') {
      std::cout << std::endl << "Go to frame: ";
      int goto_frame;
      std::cin >> goto_frame;
      if (goto_frame >= 0 && goto_frame < video_frame_num) {
        vid_idx = goto_frame;
        inputVideo.set(CV_CAP_PROP_POS_FRAMES, vid_idx);
      }
    }

    time_io += timer2 - timer1;
    time_imgproc += timer3 - timer2;
    time_nn += timer4 - timer3;
    time_stitching += timer5 - timer4;
    time_rendering += timer6 - timer5;
    if (vid_idx % 30 == 0) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(3)
         << "io: " << time_io.count() / accumulated_frames << "ms, "
         << "imgproc: " << time_imgproc.count() / accumulated_frames << "ms, "
         << "nn: " << time_nn.count() / accumulated_frames << "ms, "
         << "stitching: " << time_stitching.count() / accumulated_frames
         << "ms, "
         << "rendering: " << time_rendering.count() / accumulated_frames
         << "ms";
      std::cout << '\r' << ss.str() << std::flush;
    }
  }

  std::cout << std::endl;
  session->Close();

  return 0;
}

int test_images(const Args &args,
                const std::unique_ptr<tensorflow::Session> &session) {
  Tensor input_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({args.img_height, args.img_width, 3}));

  std::set<int> eval_class_ids;
  if (!args.eval_classes.empty()) {
    std::vector<std::string> tokens;
    boost::split(tokens, args.eval_classes, boost::is_any_of(","));
    for (const auto &token : tokens) {
      eval_class_ids.insert(std::stoi(token));
    }
  }

  std::map<int, int> id_map;
  if (!args.class_id_mapping.empty()) {
    std::vector<string> tokens;
    boost::split(tokens, args.class_id_mapping, boost::is_any_of(","));
    for (const auto &pair : tokens) {
      std::vector<string> kvp;
      boost::split(kvp, pair, boost::is_any_of(":"));
      id_map.insert(
          std::make_pair<int, int>(std::stoi(kvp[0]), std::stoi(kvp[1])));
    }
  }

  std::ifstream fin(args.eval_list);
  const string out_path =
      args.output.empty() ? "/tmp/detection_result" : args.output;
  std::ofstream fout(out_path);
  fout << "{" << std::endl;

  int img_count = 0;
  for (std::string line; std::getline(fin, line);) {
    std::cout << '\r' << line << std::flush;
    Mat src_img = imread(line, IMREAD_COLOR);
    if (src_img.empty()) {
      std::cout << std::endl;
      LOG(ERROR) << "Can not open image: " << line;
      continue;
    }

    TimePoint timer3;
    TimePoint timer4;
    std::vector<tensorbox::Rect> out;
    Mat input = Mat::zeros(args.img_height, args.img_width, CV_8UC3);
    float true_scale = min(args.img_width / float(src_img.cols),
                           args.img_height / float(src_img.rows));
    if (true_scale >= 1.0) {
      true_scale = 1.0;
    }
    run_detection(args, true_scale, src_img, session, input_tensor, input,
                  timer3, timer4, out);

    if (out.size() == 0) {
      continue;
    }

    if (img_count > 0) {
      fout << "," << std::endl;
    }

    boost::filesystem::path p(line);
    fout << p.filename() << ":" << std::endl;

    fout << "[" << std::endl;
    int rect_count = 0;
    for (const auto &rect : out) {
      if (rect.confidence_ < args.conf_threshold ||
          (!eval_class_ids.empty() &&
           eval_class_ids.find(rect.class_id_) == eval_class_ids.end())) {
        continue;
      }

      float x1 = (rect.cx_ - rect.width_ / 2) / (src_img.cols * true_scale);
      float y1 = (rect.cy_ - rect.height_ / 2) / (src_img.rows * true_scale);
      float x2 = (rect.cx_ + rect.width_ / 2) / (src_img.cols * true_scale);
      float y2 = (rect.cy_ + rect.height_ / 2) / (src_img.rows * true_scale);

      if (rect_count > 0) {
        fout << "," << std::endl;
      }

      const int id = (id_map.find(rect.class_id_) != id_map.end())
                         ? id_map[rect.class_id_]
                         : rect.class_id_;
      fout << "\t[" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ", "
           << id << ", " << rect.confidence_ << "]";
      rect_count++;
    }
    fout << std::endl << "]";

    img_count++;
  }

  fout << std::endl << "}" << std::endl;
  return 0;
}

// -------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  Args args;
  /*
  
  string graph = "/home/xpli/data/xpli/code/tf-projects/tensorflow/models/frozen_ucar_3_class_slim_inceptionv2_balanced_sample_640x384_2016_11_09_2016_11_13_23.55_700000.pb";
  int32 num_classes = 4;  
  string output = "/home/xpli/data/xpli/code/tf-projects/tensorflow/tmp";
  int32 num_boxes = 5;
  int32 region_size = 16;
  float conf = 0;
  int32 img_height = 384;
  int32 img_weidth = 640;
  string eval_list = "/home/xpli/data/xpli/code/cafferoot/caffe_partiallabel/data/ucar/video_list.txt";
  int gpu= 0;
  */
  
  std::vector<Flag> flag_list = {
  	Flag("video", &args.video_path, "video path"),
    Flag("graph", &args.graph_path, "graph path"),
    Flag("ui", &args.show_img, "show image"),
    Flag("dump", &args.dump_result, "dump result"),
    Flag("img_width", &args.img_width, "image width"),
    Flag("img_height", &args.img_height, "image hight"),
    Flag("scale", &args.scale, "scale"),
    Flag("conf", &args.conf, "confidence"),
    Flag("inter", &args.inter, "inter"),
    Flag("num_classes", &args.num_classes, "number of classes to detect"),
    Flag("gpu", &args.gpu, "the gpus to use"),
    Flag("class_names", &args.class_names, "class names"),
    Flag("region_size", &args.region_size, "test"),
    Flag("input_layer", &args.input_layer, "test"),
    Flag("output_layers", &args.output_layers,"test"),
    Flag("num_boxes", &args.num_boxes, "test"),
    Flag("eval_list", &args.eval_list, "test"),
    Flag("eval_classes", &args.eval_classes, "test"),
    Flag("class_id_mapping", &args.class_id_mapping, "test"),
    Flag("shape_dim", &args.shape_dim, "test"),
    Flag("output", &args.output, "test"),
    Flag("training_config", &args.training_config, "test")
      
  };
  
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);

  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = load_graph(args.graph_path, args.gpu, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  LOG(INFO) << "Graph loaded: " << args.graph_path;

  args.grid_width = args.img_width / args.region_size;
  args.grid_height = args.img_height / args.region_size;
  args.conf_threshold = std::stof(args.conf);
  args.scale_param = std::stof(args.scale);
  args.interpolation = (args.inter == "linear") ? INTER_LINEAR : INTER_CUBIC;
  boost::split(args.output_layer_tokens, args.output_layers,
               boost::is_any_of(","));

  namespace fs = boost::filesystem;
  if (fs::exists(args.eval_list) && fs::is_regular_file(args.eval_list)) {
    return test_images(args, session);
  }

  if (fs::exists(args.video_path) && fs::is_regular_file(args.video_path)) {
    return test_video(args, session);
  }

  LOG(ERROR) << "Invalid video_path or eval_list.";
  return -1;
}

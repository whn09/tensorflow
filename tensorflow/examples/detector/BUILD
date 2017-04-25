# Description:
#   TensorFlow C++ inference example for labeling images.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_binary(
    name = "detector",
    srcs = [
        "detector.cc",
        "hungarian.h",
        "hungarian.cc",
        "stitch_rects.h",
        "stitch_rects.cc",
    ],
    linkopts = [
      "-fopenmp",
      "-lm",
    ],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "@opencv//:opencv",
        "@boost_archive//:boost",
        "@boost_archive//:filesystem",
        "@boost_archive//:property_tree",
    ],
)

cc_binary(
    name = "detector_writevideo",
    srcs = [
        "detector_writevideo.cc",
        "hungarian.h",
        "hungarian.cc",
        "stitch_rects.h",
        "stitch_rects.cc",
    ],
    linkopts = [
      "-fopenmp",
      "-lm",
    ],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "@opencv//:opencv",
        "@boost_archive//:boost",
        "@boost_archive//:filesystem",
        "@boost_archive//:property_tree",
    ],
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "if_android",
)

LINKER_SCRIPT = "//tensorflow/contrib/android:jni/version_script.lds"

exports_files([
    "jni/version_script.lds",
])

cc_binary(
    name = "libtensorbox_detector.so",
    srcs = [
        "tensorbox_detector.cc",
        "hungarian.h",
        "hungarian.cc",
        "stitch_rects.h",
        "stitch_rects.cc",
    ],
    copts = tf_copts(),
    linkopts = [
      "-fopenmp",
      "-landroid",
      "-ljnigraphics",
      "-llog",
      "-lm",
      "-z defs",
      "-s",
      "-Wl,--version-script",  # This line must be directly followed by LINKER_SCRIPT.
      LINKER_SCRIPT,
    ],
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "@opencv//:opencv",
        "@boost_archive//:boost",
        "@boost_archive//:filesystem",
        "@boost_archive//:property_tree",
        "//tensorflow/core:android_tensorflow_lib",
        LINKER_SCRIPT,
    ],
)

cc_binary(
    name = "vis",
    srcs = [
        "vis.cc",
        "hungarian.h",
        "hungarian.cc",
        "stitch_rects.h",
        "stitch_rects.cc",
    ],
    linkopts = [
      "-fopenmp",
      "-lm",
    ],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "@opencv//:opencv",
    ],
)


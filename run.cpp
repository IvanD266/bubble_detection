// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolov8-cls torchscript
//      yolo export model=yolov8n-cls.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolov8n-cls.torchscript
// 4. now you get ncnn model files
//      yolov8n_cls.ncnn.param
//      yolov8n_cls.ncnn.bin

#include "ncnn/net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

struct Object
{
    int label;
    float prob;
};

static void get_topk(const ncnn::Mat& cls_scores, int topk, std::vector<Object>& objects)
{
    // partial sort topk with index
    int size = cls_scores.w;
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    objects.resize(topk);
    for (int i = 0; i < topk; i++)
    {
        objects[i].label = vec[i].second;
        objects[i].prob = vec[i].first;
    }
}

static int detect_yolov8_cls(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov8;

    yolov8.opt.use_vulkan_compute = false;
    // yolov8.opt.use_bf16_storage = true;

    // https://github.com/nihui/ncnn-android-yolov8/tree/master/app/src/main/assets
    yolov8.load_param("model.ncnn.param");
    yolov8.load_model("model.ncnn.bin");
    // yolov8.load_param("yolov8s_cls.ncnn.param");
    // yolov8.load_model("yolov8s_cls.ncnn.bin");
    // yolov8.load_param("yolov8m_cls.ncnn.param");
    // yolov8.load_model("yolov8m_cls.ncnn.bin");

    const int target_size = 224;
    const int topk = 5;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // letterbox pad to target_size rectangle
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    // return top-5
    get_topk(out, topk, objects);

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
    	"bubble"

    };

    cv::Mat image = bgr.clone();

    int y_offset = 0;
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f\n", obj.label, obj.prob);

        char text[256];
        sprintf(text, "%4.1f%% %s", obj.prob * 100, class_names[obj.label]);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = 0;
        int y = y_offset;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        y_offset += label_size.height;
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main()
{
    cv::VideoCapture cap(0); // 0 = default camera

    if (!cap.isOpened())
    {
        fprintf(stderr, "Could not open camera\n");
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame; // capture frame

        if (frame.empty())
        {
            fprintf(stderr, "Empty frame\n");
            break;
        }

        std::vector<Object> objects;
        detect_yolov8_cls(frame, objects);

        // draw results on frame instead of static image
        cv::Mat display = frame.clone();

        static const char* class_names[] = { /* keep your full list here */ };

        int y_offset = 0;
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            char text[256];
            sprintf(text, "%4.1f%% %s", obj.prob * 100, class_names[obj.label]);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = 0;
            int y = y_offset;

            cv::rectangle(display,
                          cv::Rect(cv::Point(x, y),
                          cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(display, text,
                        cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 0, 0));

            y_offset += label_size.height;
        }

        cv::imshow("YOLOv8 Classification", display);

        // press ESC to exit
        if (cv::waitKey(1) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

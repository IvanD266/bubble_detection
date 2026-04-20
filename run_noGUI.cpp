// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause
// Text-only bubble detection with FPS measurement

#include "ncnn/net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>  // Only video capture, no GUI
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <numeric>
#include <deque>
#include <csignal>

// ============ CONFIGURATION ============
const int BUBBLE_CLASS_ID = 0;          
const float BUBBLE_THRESHOLD = 0.5f;    // Confidence threshold (0.0-1.0)
const int FPS_WINDOW = 10;              // Frames for smoothed FPS average
const int PRINT_INTERVAL = 1;           // Output every N frames (1 = every frame)
// =======================================

ncnn::Net yolov8;
volatile sig_atomic_t running = 1;

// Handle Ctrl+C gracefully
void signal_handler(int) { running = 0; }

static void softmax(const float* src, float* dst, int n)
{
    float maxv = src[0];
    for (int i = 1; i < n; i++)
        maxv = std::max(maxv, src[i]);

    float sum = 0.f;
    for (int i = 0; i < n; i++)
    {
        dst[i] = expf(src[i] - maxv);
        sum += dst[i];
    }
    for (int i = 0; i < n; i++)
        dst[i] /= sum;
}

static float get_class_prob(const ncnn::Mat& cls_scores, int target_class)
{
    int size = cls_scores.total();
    if (target_class < 0 || target_class >= size)
        return 0.f;

    std::vector<float> probs(size);
    std::vector<float> soft(size);

    for (int i = 0; i < size; i++)
        probs[i] = cls_scores[i];

    softmax(probs.data(), soft.data(), size);
    return soft[target_class];
}

static int detect_bubble(const cv::Mat& bgr, float& bubble_prob)
{
    yolov8.opt.use_vulkan_compute = false;
    yolov8.opt.num_threads = 4;
    yolov8.opt.use_bf16_storage = true;

    // Load model once
    static bool model_loaded = false;
    if (!model_loaded)
    {
        int ret_param = yolov8.load_param("model.ncnn.param");
        int ret_model = yolov8.load_model("model.ncnn.bin");
        if (ret_param != 0 || ret_model != 0)
        {
            fprintf(stderr, "ERROR: Failed to load model files\n");
            return -1;
        }
        model_loaded = true;
    }

    const int target_size = 224;
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // Letterbox resize
    int w = img_w, h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // Pad to target_size
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 
                          hpad/2, hpad - hpad/2, 
                          wpad/2, wpad - wpad/2, 
                          ncnn::BORDER_CONSTANT, 114.f);

    // Normalize
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // Inference
    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("in0", in_pad);
    ncnn::Mat out;
    ex.extract("out0", out);

    bubble_prob = get_class_prob(out, BUBBLE_CLASS_ID);
    return 0;
}

int main(int argc, char** argv)
{
    signal(SIGINT, signal_handler);  // Handle Ctrl+C
    signal(SIGTERM, signal_handler);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        fprintf(stderr, "ERROR: Cannot open camera\n");
        return -1;
    }

    // Optional: set resolution (comment out for default)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // FPS tracking
    std::deque<float> fps_history;
    auto t_last = std::chrono::high_resolution_clock::now();
    
    int frame_id = 0;
    float bubble_prob = 0.f;

    // Header (to stderr so it doesn't interfere with parsed output)
    fprintf(stderr, "=== Bubble Detection (Text Mode) ===\n");
    fprintf(stderr, "Config: class_id=%d threshold=%.2f fps_window=%d\n", 
            BUBBLE_CLASS_ID, BUBBLE_THRESHOLD, FPS_WINDOW);
    fprintf(stderr, "Output format: FRAME:<id>|BUBBLE:<prob>|FPS:<fps>|INF:<ms>ms\n");
    fprintf(stderr, "Press Ctrl+C to stop\n\n");
    fflush(stderr);

    while (running)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Run inference
        if (detect_bubble(frame, bubble_prob) != 0) {
            fprintf(stderr, "ERROR: Inference failed\n");
            break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // FPS calculation
        auto t_now = std::chrono::high_resolution_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_now - t_last).count();
        t_last = t_now;
        
        if (frame_ms > 0) {
            fps_history.push_back(1000.f / frame_ms);
            if (fps_history.size() > FPS_WINDOW)
                fps_history.pop_front();
        }
        float avg_fps = fps_history.empty() ? 0.f : 
                       std::accumulate(fps_history.begin(), fps_history.end(), 0.f) / fps_history.size();

        // Output result (to stdout for easy piping/parsing)
        if (frame_id % PRINT_INTERVAL == 0) {
            // Machine-parseable format
            printf("FRAME:%d|BUBBLE:%.4f|FPS:%.2f|INF:%.2fms\n", 
                   frame_id, bubble_prob, avg_fps, infer_ms);
            fflush(stdout);  // Ensure immediate output
        }

        frame_id++;
    }

    // Summary to stderr
    fprintf(stderr, "\n=== Stopped ===\n");
    fprintf(stderr, "Total frames processed: %d\n", frame_id);
    
    return 0;
}

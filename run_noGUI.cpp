// run_noGUI.cpp
#include "ncnn/net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>  // Required for cv::imwrite
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <numeric>
#include <deque>
#include <csignal>
#include <sys/stat.h>    // For mkdir
#include <string.h>
#include <cmath>

// ============ CONFIGURATION ============
const int BUBBLE_CLASS_ID = 0;          // Class index for "bubble"
const float BUBBLE_THRESHOLD = 0.01f;    // Confidence threshold (0.0-1.0)
const int FPS_WINDOW = 10;              // Frames for smoothed FPS average
const int PRINT_INTERVAL = 1;           // Output every N frames
const char* OUTPUT_DIR = "detected_bubbles";
// =======================================

ncnn::Net yolov8;
volatile sig_atomic_t running = 1;

void signal_handler(int) { running = 0; }

static void softmax(const float* src, float* dst, int n)
{
    float maxv = src[0];
    for (int i = 1; i < n; i++) maxv = std::max(maxv, src[i]);
    float sum = 0.f;
    for (int i = 0; i < n; i++) { dst[i] = expf(src[i] - maxv); sum += dst[i]; }
    for (int i = 0; i < n; i++) dst[i] /= sum;
}

#include <cmath> // Add this if missing

static float get_class_prob(const ncnn::Mat& cls_scores, int target_class)
{
    int num_classes = cls_scores.total();
    if (num_classes == 0) return 0.0f;

    // 1. Flatten tensor to 1D to safely handle any output shape (1xC, Cx1, 1x1xC, etc.)
    ncnn::Mat flat = cls_scores.reshape(num_classes);

    // 2. Copy logits & sanitize extreme/invalid values
    std::vector<float> logits(num_classes);
    for (int i = 0; i < num_classes; i++) {
        float val = flat[i];
        // Clamp to prevent expf() overflow -> inf -> NaN
        if (std::isinf(val) || std::isnan(val) || val > 50.f) val = 50.f;
        if (val < -50.f) val = -50.f;
        logits[i] = val;
    }

    // 3. Numerically stable softmax
    float max_val = logits[0];
    for (int i = 1; i < num_classes; i++)
        max_val = std::max(max_val, logits[i]);

    float sum_exp = 0.0f;
    std::vector<float> probs(num_classes);
    for (int i = 0; i < num_classes; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum_exp += probs[i];
    }

    if (sum_exp <= 1e-6f) return 0.0f; // Prevent 0/0 division

    for (int i = 0; i < num_classes; i++)
        probs[i] /= sum_exp;

    // 4. Return target class probability
    if (target_class >= 0 && target_class < num_classes)
        return probs[target_class];

    return 0.0f;
}
static int detect_bubble(const cv::Mat& bgr, float& bubble_prob)
{
    yolov8.opt.use_vulkan_compute = false;
    yolov8.opt.num_threads = 4;
    yolov8.opt.use_bf16_storage = true;

    static bool model_loaded = false;
    if (!model_loaded) {
        int r1 = yolov8.load_param("model.ncnn.param");
        int r2 = yolov8.load_model("model.ncnn.bin");
        if (r1 != 0 || r2 != 0) { fprintf(stderr, "ERROR: Failed to load model\n"); return -1; }
        model_loaded = true;
    }

    const int target_size = 224;
    int w = bgr.cols, h = bgr.rows;
    float scale = 1.f;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = h * scale; }
    else       { scale = (float)target_size / h; h = target_size; w = w * scale; }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, w, h);
    int wpad = target_size - w, hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad/2, hpad - hpad/2, wpad/2, wpad - wpad/2, ncnn::BORDER_CONSTANT, 114.f);
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

ncnn::Extractor ex = yolov8.create_extractor();
ex.input("in0", in_pad);
ncnn::Mat out;
    ex.extract("out0", out);

    // out shape: w=8400, h=5, c=1 → 8400 proposals × 5 values each
    // Column index 4 = confidence logits for class 0 (bubble)
    const float* conf_logits = out.channel(0).row(4); 
    int num_proposals = out.w; // 8400
    
    float max_conf = 0.f;
    for (int i = 0; i < num_proposals; i++) {
        float logit = conf_logits[i];
        // Apply sigmoid: conf = 1 / (1 + e^-logit)
        float conf = 1.f / (1.f + expf(-logit));
        if (conf > max_conf) max_conf = conf;
    }

    bubble_prob = max_conf;
    return 0;
}

int main(int argc, char** argv)
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create output directory
    mkdir(OUTPUT_DIR, 0755);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { fprintf(stderr, "ERROR: Cannot open camera\n"); return -1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::deque<float> fps_history;
    auto t_last = std::chrono::high_resolution_clock::now();
    int frame_id = 0;
    float bubble_prob = 0.f;
    int saved_count = 0;

    fprintf(stderr, "=== Bubble Detection (Text Mode + Frame Save) ===\n");
    fprintf(stderr, "Output dir: %s | Threshold: %.2f\n\n", OUTPUT_DIR, BUBBLE_THRESHOLD);
    fflush(stderr);

    while (running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        auto t0 = std::chrono::high_resolution_clock::now();
        if (detect_bubble(frame, bubble_prob) != 0) break;
        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        auto t_now = std::chrono::high_resolution_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_now - t_last).count();
        t_last = t_now;
        if (frame_ms > 0) {
            fps_history.push_back(1000.f / frame_ms);
            if (fps_history.size() > FPS_WINDOW) fps_history.pop_front();
        }
        float avg_fps = fps_history.empty() ? 0.f : std::accumulate(fps_history.begin(), fps_history.end(), 0.f) / fps_history.size();

        bool is_bubble = (bubble_prob >= BUBBLE_THRESHOLD);

        // Save frame if detected
        if (is_bubble) {
            char filepath[128];
            sprintf(filepath, "%s/bubble_%04d_%.1f.jpg", OUTPUT_DIR, frame_id, bubble_prob * 100.f);
            cv::imwrite(filepath, frame);
            saved_count++;
        }

        // Output text line
        if (frame_id % PRINT_INTERVAL == 0) {
            printf("FRAME:%d|BUBBLE:%.4f|FPS:%.2f|INF:%.2fms|SAVED:%s\n", 
                   frame_id, bubble_prob, avg_fps, infer_ms, is_bubble ? "YES" : "NO");
            fflush(stdout);
        }

        frame_id++;
    }

    fprintf(stderr, "\n=== Stopped ===\nTotal frames: %d | Saved: %d\n", frame_id, saved_count);
    return 0;
}

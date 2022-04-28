#include <Arduino.h>
#include <PDM.h>
#include <stdint.h>

#include <TensorFlowLite.h>

#include "main_functions.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;


constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = true; // Set this to true to see e.g. features generated from the raw signal

#define EI_DSP_MATRIX(name, ...) matrix_t name(__VA_ARGS__); if (!name.buffer) { -1; }
#define EI_DSP_MATRIX_B(name, ...) matrix_t name(__VA_ARGS__); if (!name.buffer) { -1; }

const char* classifier_inferencing_categories[] = { "Bee", "notBee" };

uint8_t dsp_config_55_axes[] = { 0 };
const uint32_t dsp_config_55_axes_size = 1;
dsp_config_mfcc_t dsp_config_55 = {
    2,
    1,
    13,
    0.032f,
    0.032f,
    40,
    256,
    101,
    300,
    0,
    0.98f,
    1
};

const size_t dsp_blocks_size = 1;
model_dsp_t dsp_blocks[dsp_blocks_size] = {
    { // DSP block 55
        806,
        &extract_mfcc_features,
        (void*)&dsp_config_55,
        dsp_config_55_axes,
        dsp_config_55_axes_size
    }
};

/**
 * Size of a matrix
 */
typedef struct {
    uint32_t rows;
    uint32_t cols;
} matrix_size_t;

typedef struct {
    uint16_t implementation_version;
    int axes;
    int num_cepstral;
    float frame_length;
    float frame_stride;
    int num_filters;
    int fft_length;
    int win_size;
    int low_frequency;
    int high_frequency;
    float pre_cof;
    int pre_shift;
} dsp_config_mfcc_t;

typedef struct {
    size_t n_output_features;
    int (*extract_fn)(signal_t *signal, matrix_t *output_matrix, void *config, const float frequency);
    void *config;
    uint8_t *axes;
    size_t axes_size;
} model_dsp_t;

typedef struct ei_matrix {
    float *buffer;
    uint32_t rows;
    uint32_t cols;
    bool buffer_managed_by_me;

    /**
     * Create a new matrix
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param a_buffer Buffer, if not provided we'll alloc on the heap
     */
    ei_matrix(
        uint32_t n_rows,
        uint32_t n_cols,
        float *a_buffer = NULL
        )
    {
        if (a_buffer) {
            buffer = a_buffer;
            buffer_managed_by_me = false;
        }
        else {
            buffer = (float*)calloc(n_rows * n_cols * sizeof(float), 1);
            buffer_managed_by_me = true;
        }
        rows = n_rows;
        cols = n_cols;

        if (!a_buffer) {
        }
    }
    ~ei_matrix() {
        if (buffer && buffer_managed_by_me) {
            free(buffer);
        }
    }
} matrix_t;

/**
 * Sensor signal structure
 */
typedef struct ei_signal_t {
    /**
     * A function to retrieve part of the sensor signal
     * No bytes will be requested outside of the `total_length`.
     * @param offset The offset in the signal
     * @param length The total length of the signal
     * @param out_ptr An out buffer to set the signal data
     */

    mbed::Callback<int(size_t offset, size_t length, float *out_ptr)> get_data;
    size_t total_length;
} signal_t;

typedef struct {
    const char *label;
    float value;
} result_classification_t;

typedef struct {
    int sampling;
    int dsp;
    int classification;
    int anomaly;
    int64_t dsp_us;
    int64_t classification_us;
    int64_t anomaly_us;
} result_timing_t;

typedef struct {
    result_classification_t classification[2];
    float anomaly;
    result_timing_t timing;
    int32_t label_detected;
} result_t;

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

   /**
   * @brief 32-bit floating-point type definition.
   */
  typedef float float32_t;

 /**
  @par
  Example code for Floating-point Twiddle factors Generation:
  @par
  <pre>for (i = 0; i< N/; i++)
  {
 	twiddleCoef[2*i]   = cos(i * 2*PI/(float)N);
 	twiddleCoef[2*i+1] = sin(i * 2*PI/(float)N);
  } </pre>
  @par
  where N = 128, PI = 3.14159265358979
  @par
  Cos and Sin values are in interleaved fashion
*/
const float32_t twiddleCoef_128[256] = {
    1.000000000f,  0.000000000f,
    0.998795456f,  0.049067674f,
    0.995184727f,  0.098017140f,
    0.989176510f,  0.146730474f,
    0.980785280f,  0.195090322f,
    0.970031253f,  0.242980180f,
    0.956940336f,  0.290284677f,
    0.941544065f,  0.336889853f,
    0.923879533f,  0.382683432f,
    0.903989293f,  0.427555093f,
    0.881921264f,  0.471396737f,
    0.857728610f,  0.514102744f,
    0.831469612f,  0.555570233f,
    0.803207531f,  0.595699304f,
    0.773010453f,  0.634393284f,
    0.740951125f,  0.671558955f,
    0.707106781f,  0.707106781f,
    0.671558955f,  0.740951125f,
    0.634393284f,  0.773010453f,
    0.595699304f,  0.803207531f,
    0.555570233f,  0.831469612f,
    0.514102744f,  0.857728610f,
    0.471396737f,  0.881921264f,
    0.427555093f,  0.903989293f,
    0.382683432f,  0.923879533f,
    0.336889853f,  0.941544065f,
    0.290284677f,  0.956940336f,
    0.242980180f,  0.970031253f,
    0.195090322f,  0.980785280f,
    0.146730474f,  0.989176510f,
    0.098017140f,  0.995184727f,
    0.049067674f,  0.998795456f,
    0.000000000f,  1.000000000f,
   -0.049067674f,  0.998795456f,
   -0.098017140f,  0.995184727f,
   -0.146730474f,  0.989176510f,
   -0.195090322f,  0.980785280f,
   -0.242980180f,  0.970031253f,
   -0.290284677f,  0.956940336f,
   -0.336889853f,  0.941544065f,
   -0.382683432f,  0.923879533f,
   -0.427555093f,  0.903989293f,
   -0.471396737f,  0.881921264f,
   -0.514102744f,  0.857728610f,
   -0.555570233f,  0.831469612f,
   -0.595699304f,  0.803207531f,
   -0.634393284f,  0.773010453f,
   -0.671558955f,  0.740951125f,
   -0.707106781f,  0.707106781f,
   -0.740951125f,  0.671558955f,
   -0.773010453f,  0.634393284f,
   -0.803207531f,  0.595699304f,
   -0.831469612f,  0.555570233f,
   -0.857728610f,  0.514102744f,
   -0.881921264f,  0.471396737f,
   -0.903989293f,  0.427555093f,
   -0.923879533f,  0.382683432f,
   -0.941544065f,  0.336889853f,
   -0.956940336f,  0.290284677f,
   -0.970031253f,  0.242980180f,
   -0.980785280f,  0.195090322f,
   -0.989176510f,  0.146730474f,
   -0.995184727f,  0.098017140f,
   -0.998795456f,  0.049067674f,
   -1.000000000f,  0.000000000f,
   -0.998795456f, -0.049067674f,
   -0.995184727f, -0.098017140f,
   -0.989176510f, -0.146730474f,
   -0.980785280f, -0.195090322f,
   -0.970031253f, -0.242980180f,
   -0.956940336f, -0.290284677f,
   -0.941544065f, -0.336889853f,
   -0.923879533f, -0.382683432f,
   -0.903989293f, -0.427555093f,
   -0.881921264f, -0.471396737f,
   -0.857728610f, -0.514102744f,
   -0.831469612f, -0.555570233f,
   -0.803207531f, -0.595699304f,
   -0.773010453f, -0.634393284f,
   -0.740951125f, -0.671558955f,
   -0.707106781f, -0.707106781f,
   -0.671558955f, -0.740951125f,
   -0.634393284f, -0.773010453f,
   -0.595699304f, -0.803207531f,
   -0.555570233f, -0.831469612f,
   -0.514102744f, -0.857728610f,
   -0.471396737f, -0.881921264f,
   -0.427555093f, -0.903989293f,
   -0.382683432f, -0.923879533f,
   -0.336889853f, -0.941544065f,
   -0.290284677f, -0.956940336f,
   -0.242980180f, -0.970031253f,
   -0.195090322f, -0.980785280f,
   -0.146730474f, -0.989176510f,
   -0.098017140f, -0.995184727f,
   -0.049067674f, -0.998795456f,
   -0.000000000f, -1.000000000f,
    0.049067674f, -0.998795456f,
    0.098017140f, -0.995184727f,
    0.146730474f, -0.989176510f,
    0.195090322f, -0.980785280f,
    0.242980180f, -0.970031253f,
    0.290284677f, -0.956940336f,
    0.336889853f, -0.941544065f,
    0.382683432f, -0.923879533f,
    0.427555093f, -0.903989293f,
    0.471396737f, -0.881921264f,
    0.514102744f, -0.857728610f,
    0.555570233f, -0.831469612f,
    0.595699304f, -0.803207531f,
    0.634393284f, -0.773010453f,
    0.671558955f, -0.740951125f,
    0.707106781f, -0.707106781f,
    0.740951125f, -0.671558955f,
    0.773010453f, -0.634393284f,
    0.803207531f, -0.595699304f,
    0.831469612f, -0.555570233f,
    0.857728610f, -0.514102744f,
    0.881921264f, -0.471396737f,
    0.903989293f, -0.427555093f,
    0.923879533f, -0.382683432f,
    0.941544065f, -0.336889853f,
    0.956940336f, -0.290284677f,
    0.970031253f, -0.242980180f,
    0.980785280f, -0.195090322f,
    0.989176510f, -0.146730474f,
    0.995184727f, -0.098017140f,
    0.998795456f, -0.049067674f
};

const uint16_t armBitRevIndexTable128[((uint16_t)208)] =
{
   /* 8x2, size 208 */
   8,512, 16,64, 24,576, 32,128, 40,640, 48,192, 56,704, 64,256, 72,768,
   80,320, 88,832, 96,384, 104,896, 112,448, 120,960, 128,512, 136,520,
   144,768, 152,584, 160,520, 168,648, 176,200, 184,712, 192,264, 200,776,
   208,328, 216,840, 224,392, 232,904, 240,456, 248,968, 264,528, 272,320,
   280,592, 288,768, 296,656, 304,328, 312,720, 328,784, 344,848, 352,400,
   360,912, 368,464, 376,976, 384,576, 392,536, 400,832, 408,600, 416,584,
   424,664, 432,840, 440,728, 448,592, 456,792, 464,848, 472,856, 480,600,
   488,920, 496,856, 504,984, 520,544, 528,576, 536,608, 552,672, 560,608,
   568,736, 576,768, 584,800, 592,832, 600,864, 608,800, 616,928, 624,864,
   632,992, 648,672, 656,896, 664,928, 688,904, 696,744, 704,896, 712,808,
   720,912, 728,872, 736,928, 744,936, 752,920, 760,1000, 776,800, 784,832,
   792,864, 808,904, 816,864, 824,920, 840,864, 856,880, 872,944, 888,1008,
   904,928, 912,960, 920,992, 944,968, 952,1000, 968,992, 984,1008
};

const float32_t twiddleCoef_rfft_256[256] = {
    0.000000000f,  1.000000000f,
    0.024541229f,  0.999698819f,
    0.049067674f,  0.998795456f,
    0.073564564f,  0.997290457f,
    0.098017140f,  0.995184727f,
    0.122410675f,  0.992479535f,
    0.146730474f,  0.989176510f,
    0.170961889f,  0.985277642f,
    0.195090322f,  0.980785280f,
    0.219101240f,  0.975702130f,
    0.242980180f,  0.970031253f,
    0.266712757f,  0.963776066f,
    0.290284677f,  0.956940336f,
    0.313681740f,  0.949528181f,
    0.336889853f,  0.941544065f,
    0.359895037f,  0.932992799f,
    0.382683432f,  0.923879533f,
    0.405241314f,  0.914209756f,
    0.427555093f,  0.903989293f,
    0.449611330f,  0.893224301f,
    0.471396737f,  0.881921264f,
    0.492898192f,  0.870086991f,
    0.514102744f,  0.857728610f,
    0.534997620f,  0.844853565f,
    0.555570233f,  0.831469612f,
    0.575808191f,  0.817584813f,
    0.595699304f,  0.803207531f,
    0.615231591f,  0.788346428f,
    0.634393284f,  0.773010453f,
    0.653172843f,  0.757208847f,
    0.671558955f,  0.740951125f,
    0.689540545f,  0.724247083f,
    0.707106781f,  0.707106781f,
    0.724247083f,  0.689540545f,
    0.740951125f,  0.671558955f,
    0.757208847f,  0.653172843f,
    0.773010453f,  0.634393284f,
    0.788346428f,  0.615231591f,
    0.803207531f,  0.595699304f,
    0.817584813f,  0.575808191f,
    0.831469612f,  0.555570233f,
    0.844853565f,  0.534997620f,
    0.857728610f,  0.514102744f,
    0.870086991f,  0.492898192f,
    0.881921264f,  0.471396737f,
    0.893224301f,  0.449611330f,
    0.903989293f,  0.427555093f,
    0.914209756f,  0.405241314f,
    0.923879533f,  0.382683432f,
    0.932992799f,  0.359895037f,
    0.941544065f,  0.336889853f,
    0.949528181f,  0.313681740f,
    0.956940336f,  0.290284677f,
    0.963776066f,  0.266712757f,
    0.970031253f,  0.242980180f,
    0.975702130f,  0.219101240f,
    0.980785280f,  0.195090322f,
    0.985277642f,  0.170961889f,
    0.989176510f,  0.146730474f,
    0.992479535f,  0.122410675f,
    0.995184727f,  0.098017140f,
    0.997290457f,  0.073564564f,
    0.998795456f,  0.049067674f,
    0.999698819f,  0.024541229f,
    1.000000000f,  0.000000000f,
    0.999698819f, -0.024541229f,
    0.998795456f, -0.049067674f,
    0.997290457f, -0.073564564f,
    0.995184727f, -0.098017140f,
    0.992479535f, -0.122410675f,
    0.989176510f, -0.146730474f,
    0.985277642f, -0.170961889f,
    0.980785280f, -0.195090322f,
    0.975702130f, -0.219101240f,
    0.970031253f, -0.242980180f,
    0.963776066f, -0.266712757f,
    0.956940336f, -0.290284677f,
    0.949528181f, -0.313681740f,
    0.941544065f, -0.336889853f,
    0.932992799f, -0.359895037f,
    0.923879533f, -0.382683432f,
    0.914209756f, -0.405241314f,
    0.903989293f, -0.427555093f,
    0.893224301f, -0.449611330f,
    0.881921264f, -0.471396737f,
    0.870086991f, -0.492898192f,
    0.857728610f, -0.514102744f,
    0.844853565f, -0.534997620f,
    0.831469612f, -0.555570233f,
    0.817584813f, -0.575808191f,
    0.803207531f, -0.595699304f,
    0.788346428f, -0.615231591f,
    0.773010453f, -0.634393284f,
    0.757208847f, -0.653172843f,
    0.740951125f, -0.671558955f,
    0.724247083f, -0.689540545f,
    0.707106781f, -0.707106781f,
    0.689540545f, -0.724247083f,
    0.671558955f, -0.740951125f,
    0.653172843f, -0.757208847f,
    0.634393284f, -0.773010453f,
    0.615231591f, -0.788346428f,
    0.595699304f, -0.803207531f,
    0.575808191f, -0.817584813f,
    0.555570233f, -0.831469612f,
    0.534997620f, -0.844853565f,
    0.514102744f, -0.857728610f,
    0.492898192f, -0.870086991f,
    0.471396737f, -0.881921264f,
    0.449611330f, -0.893224301f,
    0.427555093f, -0.903989293f,
    0.405241314f, -0.914209756f,
    0.382683432f, -0.923879533f,
    0.359895037f, -0.932992799f,
    0.336889853f, -0.941544065f,
    0.313681740f, -0.949528181f,
    0.290284677f, -0.956940336f,
    0.266712757f, -0.963776066f,
    0.242980180f, -0.970031253f,
    0.219101240f, -0.975702130f,
    0.195090322f, -0.980785280f,
    0.170961889f, -0.985277642f,
    0.146730474f, -0.989176510f,
    0.122410675f, -0.992479535f,
    0.098017140f, -0.995184727f,
    0.073564564f, -0.997290457f,
    0.049067674f, -0.998795456f,
    0.024541229f, -0.999698819f
};
  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const float32_t *pTwiddle;         /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */
  } arm_cfft_instance_f32;

   const arm_cfft_instance_f32 arm_cfft_sR_f32_len128 = {
  128, twiddleCoef_128, armBitRevIndexTable128, ((uint16_t)208)
};
    
    /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const float32_t *pTwiddle;         /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */

  } arm_cfft_instance_f32;

    /**
   * @brief Instance structure for the floating-point RFFT/RIFFT function.
   */
typedef struct
  {
          arm_cfft_instance_f32 Sint;      /**< Internal CFFT structure. */
          uint16_t fftLenRFFT;             /**< length of the real sequence */
    const float32_t * pTwiddleRFFT;        /**< Twiddle factors real stage  */
  } arm_rfft_fast_instance_f32 ;

/**
 * Print wrapper around printf()
 * This is used internally to print debug information.
 */
__attribute__ ((format (printf, 1, 2)))
void serial_printf(const char *format, ...);

__attribute__((weak)) 
void serial_printf_float(float f) {
    Serial.print(f, 6);
}

class numpy {

public:
    /**
     * Roll array elements along a given axis.
     * Elements that roll beyond the last position are re-introduced at the first.
     * @param input_array
     * @param input_array_size
     * @param shift The number of places by which elements are shifted.
     * @returns EIDSP_OK if OK
     */
    static int roll(float *input_array, size_t input_array_size, int shift) {
        if (shift < 0) {
            shift = input_array_size + shift;
        }

        if (shift == 0) {
            return 0;
        }

        // so we need to allocate a buffer of the size of shift...
        EI_DSP_MATRIX(shift_matrix, 1, shift);

        // we copy from the end of the buffer into the shift buffer
        memcpy(shift_matrix.buffer, input_array + input_array_size - shift, shift * sizeof(float));

        // now we do a memmove to shift the array
        memmove(input_array + shift, input_array, (input_array_size - shift) * sizeof(float));

        // and copy the shift buffer back to the beginning of the array
        memcpy(input_array, shift_matrix.buffer, shift * sizeof(float));

        return 0;
    }

    /**
     * Scale a matrix in place
     * @param matrix
     * @param scale
     * @returns 0 if OK
     */
    static int scale(matrix_t *matrix, float scale) {
        if (scale == 1.0f) return 0;

        for (size_t ix = 0; ix < matrix->rows * matrix->cols; ix++) {
            matrix->buffer[ix] *= scale;
        }

        return 0;
    }

    /**
     * Compute the one-dimensional discrete Fourier Transform for real input.
     * This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of
     * a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).
     * @param src Source buffer
     * @param src_size Size of the source buffer
     * @param output Output buffer
     * @param output_size Size of the output buffer, should be n_fft / 2 + 1
     * @returns 0 if OK
     */
    static int rfft(const float *src, size_t src_size, float *output, size_t output_size, size_t n_fft) {
        size_t n_fft_out_features = (n_fft / 2) + 1;
        if (output_size != n_fft_out_features) {
            return -1;
        }

        // truncate if needed
        if (src_size > n_fft) {
            src_size = n_fft;
        }

        // declare input and output arrays
        EI_DSP_MATRIX(fft_input, 1, n_fft);
        if (!fft_input.buffer) {
            return -1;
        }

        // copy from src to fft_input
        memcpy(fft_input.buffer, src, src_size * sizeof(float));
        // pad to the rigth with zeros
        memset(fft_input.buffer + src_size, 0, (n_fft - src_size) * sizeof(float));

        if (n_fft != 32 && n_fft != 64 && n_fft != 128 && n_fft != 256 &&
            n_fft != 512 && n_fft != 1024 && n_fft != 2048 && n_fft != 4096) {
            int ret = software_rfft(fft_input.buffer, output, n_fft, n_fft_out_features);
            if (ret != 0) {
                return -1;
        }
        else {
            // hardware acceleration only works for the powers above...
            arm_rfft_fast_instance_f32 rfft_instance;
            int status = cmsis_rfft_init_f32(&rfft_instance, n_fft);
            if (status != 0) {
                return status;
            }

            EI_DSP_MATRIX(fft_output, 1, n_fft);
            if (!fft_output.buffer) {
                return -1;
            }

            arm_rfft_fast_f32(&rfft_instance, fft_input.buffer, fft_output.buffer, 0);

            output[0] = fft_output.buffer[0];
            output[n_fft_out_features - 1] = fft_output.buffer[1];

            size_t fft_output_buffer_ix = 2;
            for (size_t ix = 1; ix < n_fft_out_features - 1; ix += 1) {
                float rms_result;
                arm_rms_f32(fft_output.buffer + fft_output_buffer_ix, 2, &rms_result);
                output[ix] = rms_result * sqrt(2);

                fft_output_buffer_ix += 2;
            }
        }

        return 0;
    }
  
}
    private:

    
    static int software_rfft(float *fft_input, float *output, size_t n_fft, size_t n_fft_out_features) {
        kiss_fft_cpx *fft_output = (kiss_fft_cpx*)ei_dsp_malloc(n_fft_out_features * sizeof(kiss_fft_cpx));
        if (!fft_output) {
            return -1;
        }

        size_t kiss_fftr_mem_length;

        // create fftr context
        kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, 0, NULL, NULL, &kiss_fftr_mem_length);
        if (!cfg) {
            ei_dsp_free(fft_output, n_fft_out_features * sizeof(kiss_fft_cpx));
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        ei_dsp_register_alloc(kiss_fftr_mem_length, cfg);

        // execute the rfft operation
        kiss_fftr(cfg, fft_input, fft_output);

        // and write back to the output
        for (size_t ix = 0; ix < n_fft_out_features; ix++) {
            output[ix] = sqrt(pow(fft_output[ix].r, 2) + pow(fft_output[ix].i, 2));
        }

        ei_dsp_free(cfg, kiss_fftr_mem_length);
        ei_dsp_free(fft_output, n_fft_out_features * sizeof(kiss_fft_cpx));

        return EIDSP_OK;
    }

     /**
     * Initialize a CMSIS-DSP fast rfft structure
     * We do it this way as this means we can compile out fast_init calls which hints the compiler
     * to which tables can be removed
     */
    static int cmsis_rfft_init_f32(arm_rfft_fast_instance_f32 *rfft_instance, const size_t n_fft)
    {
// ARM cores (ex M55) with Helium extensions (MVEF) need special treatment (Issue 2843)

        switch (n_fft) {
            case 256: {
                arm_cfft_instance_f32 *S = &(rfft_instance->Sint);
                S->fftLen = 128U;
                S->pTwiddle = NULL;
                S->bitRevLength = arm_cfft_sR_f32_len128.bitRevLength;
                S->pBitRevTable = arm_cfft_sR_f32_len128.pBitRevTable;
                S->pTwiddle = arm_cfft_sR_f32_len128.pTwiddle;
                rfft_instance->fftLenRFFT = 256U;
                rfft_instance->pTwiddleRFFT = (float32_t *) twiddleCoef_rfft_256;
                return 0;
            }
            default:
                return -1;
        }
    }
    // more numpy code here
};


namespace speechpy {

// one stack frame returned by stack_frames
typedef struct ei_stack_frames_info {
    signal_t *signal;
    std::vector<uint32_t> *frame_ixs;
    int frame_length;

    // start_ixs is owned by us
    ~ei_stack_frames_info() {
        if (frame_ixs) {
            delete frame_ixs;
        }
    }
} stack_frames_info_t;

namespace processing {
    /**
     * Lazy Preemphasising on the signal.
     * @param signal: The input signal.
     * @param shift (int): The shift step.
     * @param cof (float): The preemphasising coefficient. 0 equals to no filtering.
     */
    class preemphasis {
public:
        preemphasis(ei_signal_t *signal, int shift, float cof, bool rescale)
            : _signal(signal), _shift(shift), _cof(cof), _rescale(rescale)
        {
            _prev_buffer = (float*)calloc(shift * sizeof(float), 1);
            _end_of_signal_buffer = (float*)calloc(shift * sizeof(float), 1);
            _next_offset_should_be = 0;

            if (shift < 0) {
                _shift = signal->total_length + shift;
            }

            if (!_prev_buffer || !_end_of_signal_buffer) return;

            // we need to get the shift bytes from the end of the buffer...
            signal->get_data(signal->total_length - shift, shift, _end_of_signal_buffer);
        }

        /**
         * Get preemphasized data from the underlying audio buffer...
         * This retrieves data from the signal then preemphasizes it.
         * @param offset Offset in the audio signal
         * @param length Length of the audio signal
         */
        int get_data(size_t offset, size_t length, float *out_buffer) {
            if (!_prev_buffer || !_end_of_signal_buffer) {
                return -1;
            }
            if (offset + length > _signal->total_length) {
                return -1;
            }

            int ret;
            if (static_cast<int32_t>(offset) - _shift >= 0) {
                ret = _signal->get_data(offset - _shift, _shift, _prev_buffer);
                if (ret != 0) {
                    return -1;
                }
            }
            // else we'll use the end_of_signal_buffer; so no need to check

            ret = _signal->get_data(offset, length, out_buffer);
            if (ret != 0) {
                return -1;
            }

            // it might be that everything is already normalized here...
            bool all_between_min_1_and_1 = true;

            // now we have the signal and we can preemphasize
            for (size_t ix = 0; ix < length; ix++) {
                float now = out_buffer[ix];

                // under shift? read from end
                if (offset + ix < static_cast<uint32_t>(_shift)) {
                    out_buffer[ix] = now - (_cof * _end_of_signal_buffer[offset + ix]);
                }
                // otherwise read from history buffer
                else {
                    out_buffer[ix] = now - (_cof * _prev_buffer[0]);
                }

                if (_rescale && all_between_min_1_and_1) {
                    if (out_buffer[ix] < -1.0f || out_buffer[ix] > 1.0f) {
                        all_between_min_1_and_1 = false;
                    }
                }

                // roll through and overwrite last element
                if (_shift != 1) {
                    numpy::roll(_prev_buffer, _shift, -1);
                }
                _prev_buffer[_shift - 1] = now;
            }

            _next_offset_should_be += length;

            // rescale from [-1 .. 1] ?
            if (_rescale && !all_between_min_1_and_1) {
                matrix_t scale_matrix(length, 1, out_buffer);
                ret = numpy::scale(&scale_matrix, 1.0f / 32768.0f);
                if (ret != 0) {
                    return -1;
                }
            }

            return 0;
        }

        ~preemphasis() {
            if (_prev_buffer) {
                free(_prev_buffer);
            }
            if (_end_of_signal_buffer) {
                free(_end_of_signal_buffer);
            }
        }

private:
        ei_signal_t *_signal;
        int _shift;
        float _cof;
        float *_prev_buffer;
        float *_end_of_signal_buffer;
        size_t _next_offset_should_be;
        bool _rescale;
    };
}

namespace processing {

    /**
     * Power spectrum of a frame
     * @param frame Row of a frame
     * @param frame_size Size of the frame
     * @param out_buffer Out buffer, size should be fft_points
     * @param out_buffer_size Buffer size
     * @param fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
     * @returns EIDSP_OK if OK
     */
    static int power_spectrum(float *frame, size_t frame_size, float *out_buffer, size_t out_buffer_size, uint16_t fft_points)
    {
        if (out_buffer_size != static_cast<size_t>(fft_points / 2 + 1)) {
            return -1;
        }

        int r = numpy::rfft(frame, frame_size, out_buffer, out_buffer_size, fft_points);
        if (r != 0) {
            return r;
        }

        for (size_t ix = 0; ix < out_buffer_size; ix++) {
            out_buffer[ix] = (1.0 / static_cast<float>(fft_points)) *
                (out_buffer[ix] * out_buffer[ix]);
        }

        return 0;
    }

    /**
     * Frame a signal into overlapping frames.
     * @param info This is both the base object and where we'll store our results.
     * @param sampling_frequency (int): The sampling frequency of the signal.
     * @param frame_length (float): The length of the frame in second.
     * @param frame_stride (float): The stride between frames.
     * @param zero_padding (bool): If the samples is not a multiple of
     *        frame_length(number of frames sample), zero padding will
     *        be done for generating last frame.
     * @returns EIDSP_OK if OK
     */
    static int stack_frames(stack_frames_info_t *info,
                            float sampling_frequency,
                            float frame_length,
                            float frame_stride,
                            bool zero_padding,
                            uint16_t version)
    {
        if (!info->signal || !info->signal->get_data || info->signal->total_length == 0) {
            return -1;
        }

        size_t length_signal = info->signal->total_length;
        int frame_sample_length;
        int length;
        if (version == 1) {
            frame_sample_length = static_cast<int>(round(static_cast<float>(sampling_frequency) * frame_length));
            frame_stride = round(static_cast<float>(sampling_frequency) * frame_stride);
            length = frame_sample_length;
        }
        else {
            frame_sample_length = static_cast<int>(ceil_unless_very_close_to_floor(static_cast<float>(sampling_frequency) * frame_length));
            float frame_stride_arg = frame_stride;
            frame_stride = ceil_unless_very_close_to_floor(static_cast<float>(sampling_frequency) * frame_stride_arg);
            length = (frame_sample_length - (int)frame_stride);
        }

        volatile int numframes;
        volatile int len_sig;

        if (zero_padding) {
            // Calculation of number of frames
            numframes = static_cast<int>(
                ceil(static_cast<float>(length_signal - length) / frame_stride));

            // Zero padding
            len_sig = static_cast<int>(static_cast<float>(numframes) * frame_stride) + frame_sample_length;

            info->signal->total_length = static_cast<size_t>(len_sig);
        }
        else {
            numframes = static_cast<int>(
                floor(static_cast<float>(length_signal - length) / frame_stride));
            len_sig = static_cast<int>(
                (static_cast<float>(numframes - 1) * frame_stride + frame_sample_length));

            info->signal->total_length = static_cast<size_t>(len_sig);
        }

        // alloc the vector on the heap, will be owned by the info struct
        std::vector<uint32_t> *frame_indices = new std::vector<uint32_t>();

        int frame_count = 0;

        for (size_t ix = 0; ix < static_cast<uint32_t>(len_sig); ix += static_cast<size_t>(frame_stride)) {
            if (++frame_count > numframes) break;

            frame_indices->push_back(ix);
        }

        info->frame_ixs = frame_indices;
        info->frame_length = frame_sample_length;

        return 0;
    }

    /**
     * frame_length is a float and can thus be off by a little bit, e.g.
     * frame_length = 0.018f actually can yield 0.018000011f
     * thus screwing up our frame calculations here...
     */
    static float ceil_unless_very_close_to_floor(float v) {
        if (v > floor(v) && v - floor(v) < 0.001f) {
            v = (floor(v));
        }
        else {
            v = (ceil(v));
        }
        return v;
    }

    /**
     * Calculate the number of stack frames for the settings provided.
     * This is needed to allocate the right buffer size for the output of f.e. the MFE
     * blocks.
     * @param signal_size: The number of frames in the signal
     * @param sampling_frequency (int): The sampling frequency of the signal.
     * @param frame_length (float): The length of the frame in second.
     * @param frame_stride (float): The stride between frames.
     * @param zero_padding (bool): If the samples is not a multiple of
     *        frame_length(number of frames sample), zero padding will
     *        be done for generating last frame.
     * @returns Number of frames required, or a negative number if an error occured
     */
    static int32_t calculate_no_of_stack_frames(
        size_t signal_size,
        uint32_t sampling_frequency,
        float frame_length,
        float frame_stride,
        bool zero_padding,
        uint16_t version)
    {
        int frame_sample_length;
        int length;
        if (version == 1) {
            frame_sample_length = static_cast<int>(round(static_cast<float>(sampling_frequency) * frame_length));
            frame_stride = round(static_cast<float>(sampling_frequency) * frame_stride);
            length = frame_sample_length;
        }
        else {
            frame_sample_length = static_cast<int>(ceil_unless_very_close_to_floor(static_cast<float>(sampling_frequency) * frame_length));
            float frame_stride_arg = frame_stride;
            frame_stride = ceil_unless_very_close_to_floor(static_cast<float>(sampling_frequency) * frame_stride_arg);
            length = (frame_sample_length - (int)frame_stride);
        }

        volatile int numframes;

        if (zero_padding) {
            // Calculation of number of frames
            numframes = static_cast<int>(
                ceil(static_cast<float>(signal_size - length) / frame_stride));
        }
        else {
            numframes = static_cast<int>(
                floor(static_cast<float>(signal_size - length) / frame_stride));
        }

        return numframes;
    }

    /**
     * This function performs local cepstral mean and
     * variance normalization on a sliding window. The code assumes that
     * there is one observation per row.
     * @param features_matrix input feature matrix, will be modified in place
     * @param win_size The size of sliding window for local normalization.
     *   Default=301 which is around 3s if 100 Hz rate is
     *   considered(== 10ms frame stide)
     * @param variance_normalization If the variance normilization should
     *   be performed or not.
     * @param scale Scale output to 0..1
     * @returns 0 if OK
     */
    static int cmvnw(matrix_t *features_matrix, uint16_t win_size = 301, bool variance_normalization = false,
        bool scale = false)
    {
        if (win_size == 0) {
            return 0;
        }

        uint16_t pad_size = (win_size - 1) / 2;

        int ret;
        float *features_buffer_ptr;

        // mean & variance normalization
        EI_DSP_MATRIX(vec_pad, features_matrix->rows + (pad_size * 2), features_matrix->cols);
        if (!vec_pad.buffer) {
            return -1;
        }

        ret = numpy::pad_1d_symmetric(features_matrix, &vec_pad, pad_size, pad_size);
        if (ret != 0) {
            return -1;
        }

        EI_DSP_MATRIX(mean_matrix, vec_pad.cols, 1);
        if (!mean_matrix.buffer) {
            return -1;
        }

        EI_DSP_MATRIX(window_variance, vec_pad.cols, 1);
        if (!window_variance.buffer) {
            return -1;
        }

        for (size_t ix = 0; ix < features_matrix->rows; ix++) {
            // create a slice on the vec_pad
            EI_DSP_MATRIX_B(window, win_size, vec_pad.cols, vec_pad.buffer + (ix * vec_pad.cols));
            if (!window.buffer) {
                return -1;
            }

            ret = numpy::mean_axis0(&window, &mean_matrix);
            if (ret != 0) {
                return -1;
            }

            // subtract the mean for the features
            for (size_t fm_col = 0; fm_col < features_matrix->cols; fm_col++) {
                features_matrix->buffer[(ix * features_matrix->cols) + fm_col] =
                    features_matrix->buffer[(ix * features_matrix->cols) + fm_col] - mean_matrix.buffer[fm_col];
            }
        }

        ret = numpy::pad_1d_symmetric(features_matrix, &vec_pad, pad_size, pad_size);
        if (ret != 0) {
            return -1;
        }

        for (size_t ix = 0; ix < features_matrix->rows; ix++) {
            // create a slice on the vec_pad
            EI_DSP_MATRIX_B(window, win_size, vec_pad.cols, vec_pad.buffer + (ix * vec_pad.cols));
            if (!window.buffer) {
                return -1;
            }

            if (variance_normalization == true) {
                ret = numpy::std_axis0(&window, &window_variance);
                if (ret != 0) {
                    return -1;
                }

                features_buffer_ptr = &features_matrix->buffer[ix * vec_pad.cols];
                for (size_t col = 0; col < vec_pad.cols; col++) {
                    *(features_buffer_ptr) = (*(features_buffer_ptr)) /
                                             (window_variance.buffer[col] + 1e-10);
                    features_buffer_ptr++;
                }
            }
        }

        if (scale) {
            ret = numpy::normalize(features_matrix);
            if (ret != 0) {
                return -1;
            }
        }

        return 0;
    }
};

class feature {
public:

    
    /**
     * Compute the Mel-filterbanks. Each filter will be stored in one rows.
     * The columns correspond to fft bins.
     *
     * @param filterbanks Matrix of size num_filter * coefficients
     * @param num_filter the number of filters in the filterbank
     * @param coefficients (fftpoints//2 + 1)
     * @param sampling_freq  the samplerate of the signal we are working
     *                       with. It affects mel spacing.
     * @param low_freq lowest band edge of mel filters, default 0 Hz
     * @param high_freq highest band edge of mel filters, default samplerate / 2
     * @param output_transposed If set to true this will transpose the matrix (memory efficient).
     *                          This is more efficient than calling this function and then transposing
     *                          as the latter requires the filterbank to be allocated twice (for a short while).
     * @returns EIDSP_OK if OK
     */
    static int filterbanks(
        matrix_t *filterbanks,
        uint16_t num_filter, int coefficients, uint32_t sampling_freq,
        uint32_t low_freq, uint32_t high_freq,
        bool output_transposed = false
        )
    {
        const size_t mels_mem_size = (num_filter + 2) * sizeof(float);
        const size_t hertz_mem_size = (num_filter + 2) * sizeof(float);
        const size_t freq_index_mem_size = (num_filter + 2) * sizeof(int);

        float *mels = (float*)malloc(mels_mem_size);
        if (!mels) {
            return -1;
        }

        if (filterbanks->rows != num_filter || filterbanks->cols != static_cast<uint32_t>(coefficients)) {
            return -1;
        }

        memset(filterbanks->buffer, 0, filterbanks->rows * filterbanks->cols * sizeof(float));

        // Computing the Mel filterbank
        // converting the upper and lower frequencies to Mels.
        // num_filter + 2 is because for num_filter filterbanks we need
        // num_filter+2 point.
        numpy::linspace(
            functions::frequency_to_mel(static_cast<float>(low_freq)),
            functions::frequency_to_mel(static_cast<float>(high_freq)),
            num_filter + 2,
            mels);

        // we should convert Mels back to Hertz because the start and end-points
        // should be at the desired frequencies.
        float *hertz = (float*)malloc(hertz_mem_size);
        if (!hertz) {
            free(mels);
            return -1;
        }
        for (uint16_t ix = 0; ix < num_filter + 2; ix++) {
            hertz[ix] = functions::mel_to_frequency(mels[ix]);
            if (hertz[ix] < low_freq) {
                hertz[ix] = low_freq;
            }
            if (hertz[ix] > high_freq) {
                hertz[ix] = high_freq;
            }

            // here is a really annoying bug in Speechpy which calculates the frequency index wrong for the last bucket
            // the last 'hertz' value is not 8,000 (with sampling rate 16,000) but 7,999.999999
            // thus calculating the bucket to 64, not 65.
            // we're adjusting this here a tiny bit to ensure we have the same result
            if (ix == num_filter + 2 - 1) {
                hertz[ix] -= 0.001;
            }
        }
        free(mels);

        // The frequency resolution required to put filters at the
        // exact points calculated above should be extracted.
        //  So we should round those frequencies to the closest FFT bin.
        int *freq_index = (int*)malloc(freq_index_mem_size);
        if (!freq_index) {
            free(hertz);
            return -1;
        }
        for (uint16_t ix = 0; ix < num_filter + 2; ix++) {
            freq_index[ix] = static_cast<int>(floor((coefficients + 1) * hertz[ix] / sampling_freq));
        }
        free(hertz);

        for (size_t i = 0; i < num_filter; i++) {
            int left = freq_index[i];
            int middle = freq_index[i + 1];
            int right = freq_index[i + 2];

            EI_DSP_MATRIX(z, 1, (right - left + 1));
            if (!z.buffer) {
                free(freq_index);
                return -1;
            }
            numpy::linspace(left, right, (right - left + 1), z.buffer);
            functions::triangle(z.buffer, (right - left + 1), left, middle, right);

            // so... z now contains some values that we need to overwrite in the filterbank
            for (int zx = 0; zx < (right - left + 1); zx++) {
                size_t index = (i * filterbanks->cols) + (left + zx);

                if (output_transposed) {
                    index = ((left + zx) * filterbanks->rows) + i;
                }

                filterbanks->buffer[index] = z.buffer[zx];
            }
        }

        if (output_transposed) {
            uint16_t r = filterbanks->rows;
            filterbanks->rows = filterbanks->cols;
            filterbanks->cols = r;
        }

        free(freq_index);

        return 0;
    }

    /**
     * Calculate the buffer size for MFCC
     * @param signal_length: Length of the signal.
     * @param sampling_frequency (int): The sampling frequency of the signal.
     * @param frame_length (float): The length of the frame in second.
     * @param frame_stride (float): The stride between frames.
     * @param num_cepstral
     */
    static matrix_size_t calculate_mfcc_buffer_size(
        size_t signal_length,
        uint32_t sampling_frequency,
        float frame_length, float frame_stride, uint16_t num_cepstral,
        uint16_t version)
    {
        uint16_t rows = processing::calculate_no_of_stack_frames(
            signal_length,
            sampling_frequency,
            frame_length,
            frame_stride,
            false,
            version);
        uint16_t cols = num_cepstral;

        matrix_size_t size_matrix;
        size_matrix.rows = rows;
        size_matrix.cols = cols;
        return size_matrix;
    }

    /**
     * Calculate the buffer size for MFE
     * @param signal_length: Length of the signal.
     * @param sampling_frequency (int): The sampling frequency of the signal.
     * @param frame_length (float): The length of the frame in second.
     * @param frame_stride (float): The stride between frames.
     * @param num_filters
     */
    static matrix_size_t calculate_mfe_buffer_size(
        size_t signal_length,
        uint32_t sampling_frequency,
        float frame_length, float frame_stride, uint16_t num_filters,
        uint16_t version)
    {
        uint16_t rows = processing::calculate_no_of_stack_frames(
            signal_length,
            sampling_frequency,
            frame_length,
            frame_stride,
            false,
            version);
        uint16_t cols = num_filters;

        matrix_size_t size_matrix;
        size_matrix.rows = rows;
        size_matrix.cols = cols;
        return size_matrix;
    }

    /**
     * Compute Mel-filterbank energy features from an audio signal.
     * @param out_features Use `calculate_mfe_buffer_size` to allocate the right matrix.
     * @param out_energies A matrix in the form of Mx1 where M is the rows from `calculate_mfe_buffer_size`
     * @param signal: audio signal structure with functions to retrieve data from a signal
     * @param sampling_frequency (int): the sampling frequency of the signal
     *     we are working with.
     * @param frame_length (float): the length of each frame in seconds.
     *     Default is 0.020s
     * @param frame_stride (float): the step between successive frames in seconds.
     *     Default is 0.02s (means no overlap)
     * @param num_filters (int): the number of filters in the filterbank,
     *     default 40.
     * @param fft_length (int): number of FFT points. Default is 512.
     * @param low_frequency (int): lowest band edge of mel filters.
     *     In Hz, default is 0.
     * @param high_frequency (int): highest band edge of mel filters.
     *     In Hz, default is samplerate/2
     * @EIDSP_OK if OK
     */
    static int mfe(matrix_t *out_features, matrix_t *out_energies,
        signal_t *signal,
        uint32_t sampling_frequency,
        float frame_length, float frame_stride, uint16_t num_filters,
        uint16_t fft_length, uint32_t low_frequency, uint32_t high_frequency,
        uint16_t version
        )
    {
        int ret = 0;

        if (high_frequency == 0) {
            high_frequency = sampling_frequency / 2;
        }

        if (low_frequency == 0) {
            low_frequency = 300;
        }

        stack_frames_info_t stack_frame_info = { 0 };
        stack_frame_info.signal = signal;

        ret = processing::stack_frames(
            &stack_frame_info,
            sampling_frequency,
            frame_length,
            frame_stride,
            false,
            version
        );
        if (ret != 0) {
            return -1;
        }

        if (stack_frame_info.frame_ixs->size() != out_features->rows) {
            return -1;
        }

        if (num_filters != out_features->cols) {
            return -1;
        }

        if (stack_frame_info.frame_ixs->size() != out_energies->rows || out_energies->cols != 1) {
            return -1;
        }

        for (uint32_t i = 0; i < out_features->rows * out_features->cols; i++) {
            *(out_features->buffer + i) = 0;
        }

        uint16_t coefficients = fft_length / 2 + 1;

        // calculate the filterbanks first... preferably I would want to do the matrix multiplications
        // whenever they happen, but OK...

        EI_DSP_MATRIX(filterbanks, num_filters, coefficients);

        if (!filterbanks.buffer) {
            return -1;
        }

        ret = feature::filterbanks(
            &filterbanks, num_filters, coefficients, sampling_frequency, low_frequency, high_frequency, true);
        if (ret != 0) {
            return -1;
        }
        for (size_t ix = 0; ix < stack_frame_info.frame_ixs->size(); ix++) {
            size_t power_spectrum_frame_size = (fft_length / 2 + 1);

            EI_DSP_MATRIX(power_spectrum_frame, 1, power_spectrum_frame_size);
            if (!power_spectrum_frame.buffer) {
                return -1;
            }

            // get signal data from the audio file
            EI_DSP_MATRIX(signal_frame, 1, stack_frame_info.frame_length);

            // don't read outside of the audio buffer... we'll automatically zero pad then
            size_t signal_offset = stack_frame_info.frame_ixs->at(ix);
            size_t signal_length = stack_frame_info.frame_length;
            if (signal_offset + signal_length > stack_frame_info.signal->total_length) {
                signal_length = signal_length -
                    (stack_frame_info.signal->total_length - (signal_offset + signal_length));
            }

            ret = stack_frame_info.signal->get_data(
                signal_offset,
                signal_length,
                signal_frame.buffer
            );
            if (ret != 0) {
                return -1;
            }

            ret = processing::power_spectrum(
                signal_frame.buffer,
                stack_frame_info.frame_length,
                power_spectrum_frame.buffer,
                power_spectrum_frame_size,
                fft_length
            );

            if (ret != 0) {
                return -1;
            }

            float energy = numpy::sum(power_spectrum_frame.buffer, power_spectrum_frame_size);
            if (energy == 0) {
                energy = 1e-10;
            }

            out_energies->buffer[ix] = energy;

            // calculate the out_features directly here
            ret = numpy::dot_by_row(
                ix,
                power_spectrum_frame.buffer,
                power_spectrum_frame_size,
                &filterbanks,
                out_features
            );

            if (ret != 0) {
                return -1;
            }
        }

        functions::zero_handling(out_features);

        return 0;
    }

    /**
     * Compute MFCC features from an audio signal.
     * @param out_features Use `calculate_mfcc_buffer_size` to allocate the right matrix.
     * @param signal: audio signal structure from which to compute features.
     *     has functions to retrieve data from a signal lazily.
     * @param sampling_frequency (int): the sampling frequency of the signal
     *     we are working with.
     * @param frame_length (float): the length of each frame in seconds.
     *     Default is 0.020s
     * @param frame_stride (float): the step between successive frames in seconds.
     *     Default is 0.01s (means no overlap)
     * @param num_cepstral (int): Number of cepstral coefficients.
     * @param num_filters (int): the number of filters in the filterbank,
     *     default 40.
     * @param fft_length (int): number of FFT points. Default is 512.
     * @param low_frequency (int): lowest band edge of mel filters.
     *     In Hz, default is 0.
     * @param high_frequency (int): highest band edge of mel filters.
     *     In Hz, default is samplerate/2
     * @param dc_elimination Whether the first dc component should
     *     be eliminated or not.
     * @returns 0 if OK
     */
    static int mfcc(matrix_t *out_features, signal_t *signal,
        uint32_t sampling_frequency, float frame_length, float frame_stride,
        uint8_t num_cepstral, uint16_t num_filters, uint16_t fft_length,
        uint32_t low_frequency, uint32_t high_frequency, bool dc_elimination,
        uint16_t version)
    {
        if (out_features->cols != num_cepstral) {
            return -1;
        }

        matrix_size_t mfe_matrix_size =
            calculate_mfe_buffer_size(
                signal->total_length,
                sampling_frequency,
                frame_length,
                frame_stride,
                num_filters,
                version);

        if (out_features->rows != mfe_matrix_size.rows) {
            return -1;
        }

        int ret = 0;

        // allocate some memory for the MFE result
        EI_DSP_MATRIX(features_matrix, mfe_matrix_size.rows, mfe_matrix_size.cols);
        if (!features_matrix.buffer) {
            return -1;
        }

        EI_DSP_MATRIX(energy_matrix, mfe_matrix_size.rows, 1);
        if (!energy_matrix.buffer) {
            return -1;
        }

        ret = mfe(&features_matrix, &energy_matrix, signal,
            sampling_frequency, frame_length, frame_stride, num_filters, fft_length,
            low_frequency, high_frequency, version);
        if (ret != 0) {
            return -1;
        }

        // ok... now we need to calculate the MFCC from this...
        // first do log() over all features...
        ret = numpy::log(&features_matrix);
        if (ret != 0) {
            return -1;
        }

        // now do DST type 2
        ret = numpy::dct2(&features_matrix, 1);
        if (ret != 0) {
            return -1;
        }

        // replace first cepstral coefficient with log of frame energy for DC elimination
        if (dc_elimination) {
            for (size_t row = 0; row < features_matrix.rows; row++) {
                features_matrix.buffer[row * features_matrix.cols] = numpy::log(energy_matrix.buffer[row]);
            }
        }

        // copy to the output...
        for (size_t row = 0; row < features_matrix.rows; row++) {
            for(int i = 0; i < num_cepstral; i++) {
                *(out_features->buffer + (num_cepstral * row) + i) = *(features_matrix.buffer + (features_matrix.cols * row) + i);
            }
        }

        return 0;
    }
};
// more speechpy classes go here ...

}


static class speechpy::processing::preemphasis *preemphasis;
static int preemphasized_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    return preemphasis->get_data(offset, length, out_ptr);
}

__attribute__((unused)) int extract_mfcc_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency) {
    dsp_config_mfcc_t config = *((dsp_config_mfcc_t*)config_ptr);

    if (config.axes != 1) {
        return -1;
    }

    if((config.implementation_version == 0) || (config.implementation_version > 3)) {
        return -1;
    }

    if (signal->total_length == 0) {
        return -1;
    }

    const uint32_t frequency = static_cast<uint32_t>(sampling_frequency);

    // preemphasis class to preprocess the audio...
    class speechpy::processing::preemphasis pre(signal, config.pre_shift, config.pre_cof, false);
    preemphasis = &pre;

    signal_t preemphasized_audio_signal;
    preemphasized_audio_signal.total_length = signal->total_length;
    preemphasized_audio_signal.get_data = &preemphasized_audio_signal_get_data;

    // calculate the size of the MFCC matrix
    matrix_size_t out_matrix_size =
        speechpy::feature::calculate_mfcc_buffer_size(
            signal->total_length, frequency, config.frame_length, config.frame_stride, config.num_cepstral, config.implementation_version);
    /* Only throw size mismatch error calculated buffer doesn't fit for continuous inferencing */
    if (out_matrix_size.rows * out_matrix_size.cols > output_matrix->rows * output_matrix->cols) {
        serial_printf("out_matrix = %dx%d\n", (int)output_matrix->rows, (int)output_matrix->cols);
        serial_printf("calculated size = %dx%d\n", (int)out_matrix_size.rows, (int)out_matrix_size.cols);
        return -1;
    }

    output_matrix->rows = out_matrix_size.rows;
    output_matrix->cols = out_matrix_size.cols;

    // and run the MFCC extraction (using 32 rather than 40 filters here to optimize speed on embedded)
    int ret = speechpy::feature::mfcc(output_matrix, &preemphasized_audio_signal,
        frequency, config.frame_length, config.frame_stride, config.num_cepstral, config.num_filters, config.fft_length,
        config.low_frequency, config.high_frequency, true, config.implementation_version);
    if (ret != 0) {
        serial_printf("ERR: MFCC failed (%d)\n", ret);
        return -1;
    }

    // cepstral mean and variance normalization
    ret = speechpy::processing::cmvnw(output_matrix, config.win_size, true, false);
    if (ret != 0) {
        serial_printf("ERR: cmvnw failed (%d)\n", ret);
        return -1;
    }

    output_matrix->cols = out_matrix_size.rows * out_matrix_size.cols;
    output_matrix->rows = 1;

    return 0;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    const int16_t *input = &inference.buffer[offset];
    for (size_t ix = 0; ix < length; ix++) {
            out_ptr[ix] = (float)(input[ix]) / 32768.f;
        }

    return 0;
}

/**
 * @brief      Wait on new data
 * 
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while(inference.buf_ready == 0) {
        delay(10);
    }

    return true;
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffer);
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if(inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize(4096);

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, 16000)) {
        Serial.print("Failed to start PDM!");
        microphone_inference_end();

        return false;
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    return true;
}

extern "C" int run_classifier(signal_t *signal, result_t *result, bool debug = false)
{

//   static float buf[1000];
//     printf("Raw data: ");
//     for (size_t ix = 0; ix < 16000; ix += 1000) {
//         int r = signal->get_data(ix, 1000, buf);
//         for (size_t jx = 0; jx < 1000; jx++) {
//             printf("%.0f, ", buf[jx]);
//         }
//     printf("\n");
//     }

    memset(result, 0, sizeof(result_t));

    matrix_t features_matrix(1, 806);

    uint64_t dsp_start_us = micros();

    size_t out_features_index = 0;

    for (size_t ix = 0; ix < 1U; ix++) {
        model_dsp_t block = dsp_blocks[ix];

        if (out_features_index + block.n_output_features > 806) {
            serial_printf("ERR: Would write outside feature buffer\n");
            return -1;
        }

        matrix_t fm(1, block.n_output_features, features_matrix.buffer + out_features_index);

        if (block.axes_size != 1) {
            serial_printf("ERR: EIDSP_SIGNAL_C_FN_POINTER can only be used when all axes are selected for DSP blocks\n");
            return -1;
        }
        int ret = block.extract_fn(signal, &fm, block.config, 16000);

        if (ret != 0) {
            serial_printf("ERR: Failed to run DSP process (%d)\n", ret);
            return -1;
        }

        out_features_index += block.n_output_features;
    }

    result->timing.dsp_us = micros() - dsp_start_us;
    result->timing.dsp = (int)(result->timing.dsp_us / 1000);

    if (debug) {
        serial_printf("Features (%d ms.): ", result->timing.dsp);
        for (size_t ix = 0; ix < features_matrix.cols; ix++) {
            serial_printf_float(features_matrix.buffer[ix]);
            serial_printf(" ");
        }
        serial_printf("\n");
    }
    
    // replace this part with own tf interpreter(?)

    // return run_inference(&features_matrix, result, debug);
}


// The name of this function is important for Arduino compatibility.
void setup() {
// put your setup code here, to run once:
    Serial.begin(115200);

    Serial.println("Edge Impulse Inferencing Demo");

    // summary of inferencing settings (from model_metadata.h)
    serial_printf("Inferencing settings:\n");
    serial_printf("\tInterval: %.2f ms.\n", (float)0.0625);
    serial_printf("\tFrame size: %d\n", 32000);
    serial_printf("\tSample length: %d ms.\n", 32000 / 16); // 2 sec
    serial_printf("\tNo. of classes: %d\n", 2);

    if (microphone_inference_start(32000) == false) {
        serial_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }

  /* // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;


  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0; */
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // put your main code here, to run repeatedly:
      serial_printf("Starting inferencing in 2 seconds...\n");

    delay(2000);

    serial_printf("Recording...\n");

    bool m = microphone_inference_record();
    if (!m) {
        serial_printf("ERR: Failed to record audio...\n");
        return;
    }

    serial_printf("Recording done\n");

    signal_t signal;
    signal.total_length = 32000;
    signal.get_data = &microphone_audio_signal_get_data;
    serial_printf("%d \n", signal.get_data);
    result_t result = { 0 };

    

    int r = run_classifier(&signal, &result, debug_nn);
    if (r != 0) {
        serial_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // print the predictions
    serial_printf("Predictions ");
    serial_printf("(DSP: %d ms., Classification: %d ms.)",
        result.timing.dsp, result.timing.classification);
    serial_printf(": \n");
    for (size_t ix = 0; ix < 2; ix++) {
        serial_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
    }





  /* // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0; */
}
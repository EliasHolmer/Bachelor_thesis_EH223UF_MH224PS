#include <Arduino.h>
#include <PDM.h>
#include <stdint.h>
#include <vector>

#include <cfloat> //numpy

#include "_kiss_fft_guts.h"
// #include <TensorFlowLite.h>

#include "main_functions.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"

// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.

// namespace {
// tflite::ErrorReporter* error_reporter = nullptr;
// const tflite::Model* model = nullptr;
// tflite::MicroInterpreter* interpreter = nullptr;
// TfLiteTensor* input = nullptr;
// TfLiteTensor* output = nullptr;
// int inference_count = 0;


// constexpr int kTensorArenaSize = 2000;
// uint8_t tensor_arena[kTensorArenaSize];
// }  // namespace

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

// ---------------------------------------------- NUMPY STRUCTS/CONSTANTS START ----------------------------------------------
// ---------------------------------------------- NUMPY STRUCTS/CONSTANTS START ----------------------------------------------
#define ei_dsp_register_alloc(...) (void)0
#define M_PI 3.14159265358979323846264338327950288

typedef enum {
    DCT_NORMALIZATION_NONE,
    DCT_NORMALIZATION_ORTHO
} DCT_NORMALIZATION_MODE;

typedef struct {
    float r;
    float i;
} fft_complex_t;

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
  } arm_rfft_fast_instance_f32;



// ---------------------------------------------- NUMPY STRUCTS/CONSTANTS END ----------------------------------------------
// ---------------------------------------------- NUMPY STRUCTS/CONSTANTS END ----------------------------------------------


// ---------------------------------------------- NUMPY DEPENDENCIES (ARM FFT) START ----------------------------------------------
// ---------------------------------------------- NUMPY DEPENDENCIES (ARM FFT) START ----------------------------------------------

/**
  @brief         Processing function for the floating-point real FFT.
  @param[in]     S         points to an arm_rfft_fast_instance_f32 structure
  @param[in]     p         points to input buffer (Source buffer is modified by this function.)
  @param[in]     pOut      points to output buffer
  @param[in]     ifftFlag
                   - value = 0: RFFT
                   - value = 1: RIFFT
  @return        none
*/

void arm_rfft_fast_f32(
  const arm_rfft_fast_instance_f32 * S,
  float32_t * p,
  float32_t * pOut,
  uint8_t ifftFlag)
{
   const arm_cfft_instance_f32 * Sint = &(S->Sint);

   /* Calculation of Real FFT */
   if (ifftFlag)
   {
      /*  Real FFT compression */
      merge_rfft_f32(S, p, pOut);
      /* Complex radix-4 IFFT process */
      arm_cfft_f32( Sint, pOut, ifftFlag, 1);
   }
   else
   {
      /* Calculation of RFFT of input */
      arm_cfft_f32( Sint, p, ifftFlag, 1);

      /*  Real FFT extraction */
      stage_rfft_f32(S, p, pOut);
   }
}

void stage_rfft_f32(
  const arm_rfft_fast_instance_f32 * S,
        float32_t * p,
        float32_t * pOut)
{
        int32_t  k;                                /* Loop Counter */
        float32_t twR, twI;                         /* RFFT Twiddle coefficients */
  const float32_t * pCoeff = S->pTwiddleRFFT;       /* Points to RFFT Twiddle factors */
        float32_t *pA = p;                          /* increasing pointer */
        float32_t *pB = p;                          /* decreasing pointer */
        float32_t xAR, xAI, xBR, xBI;               /* temporary variables */
        float32_t t1a, t1b;                         /* temporary variables */
        float32_t p0, p1, p2, p3;                   /* temporary variables */


   k = (S->Sint).fftLen - 1;

   /* Pack first and last sample of the frequency domain together */

   xBR = pB[0];
   xBI = pB[1];
   xAR = pA[0];
   xAI = pA[1];

   twR = *pCoeff++ ;
   twI = *pCoeff++ ;


   // U1 = XA(1) + XB(1); % It is real
   t1a = xBR + xAR  ;

   // U2 = XB(1) - XA(1); % It is imaginary
   t1b = xBI + xAI  ;

   // real(tw * (xB - xA)) = twR * (xBR - xAR) - twI * (xBI - xAI);
   // imag(tw * (xB - xA)) = twI * (xBR - xAR) + twR * (xBI - xAI);
   *pOut++ = 0.5f * ( t1a + t1b );
   *pOut++ = 0.5f * ( t1a - t1b );

   // XA(1) = 1/2*( U1 - imag(U2) +  i*( U1 +imag(U2) ));
   pB  = p + 2*k;
   pA += 2;

   do
   {
      /*
         function X = my_split_rfft(X, ifftFlag)
         % X is a series of real numbers
         L  = length(X);
         XC = X(1:2:end) +i*X(2:2:end);
         XA = fft(XC);
         XB = conj(XA([1 end:-1:2]));
         TW = i*exp(-2*pi*i*[0:L/2-1]/L).';
         for l = 2:L/2
            XA(l) = 1/2 * (XA(l) + XB(l) + TW(l) * (XB(l) - XA(l)));
         end
         XA(1) = 1/2* (XA(1) + XB(1) + TW(1) * (XB(1) - XA(1))) + i*( 1/2*( XA(1) + XB(1) + i*( XA(1) - XB(1))));
         X = XA;
      */

      xBI = pB[1];
      xBR = pB[0];
      xAR = pA[0];
      xAI = pA[1];

      twR = *pCoeff++;
      twI = *pCoeff++;

      t1a = xBR - xAR ;
      t1b = xBI + xAI ;

      // real(tw * (xB - xA)) = twR * (xBR - xAR) - twI * (xBI - xAI);
      // imag(tw * (xB - xA)) = twI * (xBR - xAR) + twR * (xBI - xAI);
      p0 = twR * t1a;
      p1 = twI * t1a;
      p2 = twR * t1b;
      p3 = twI * t1b;

      *pOut++ = 0.5f * (xAR + xBR + p0 + p3 ); //xAR
      *pOut++ = 0.5f * (xAI - xBI + p1 - p2 ); //xAI


      pA += 2;
      pB -= 2;
      k--;
   } while (k > 0);
}

void arm_cfft_radix8by2_f32 (arm_cfft_instance_f32 * S, float32_t * p1)
{
  uint32_t    L  = S->fftLen;
  float32_t * pCol1, * pCol2, * pMid1, * pMid2;
  float32_t * p2 = p1 + L;
  const float32_t * tw = (float32_t *) S->pTwiddle;
  float32_t t1[4], t2[4], t3[4], t4[4], twR, twI;
  float32_t m0, m1, m2, m3;
  uint32_t l;

  pCol1 = p1;
  pCol2 = p2;

  /* Define new length */
  L >>= 1;

  /* Initialize mid pointers */
  pMid1 = p1 + L;
  pMid2 = p2 + L;

  /* do two dot Fourier transform */
  for (l = L >> 2; l > 0; l-- )
  {
    t1[0] = p1[0];
    t1[1] = p1[1];
    t1[2] = p1[2];
    t1[3] = p1[3];

    t2[0] = p2[0];
    t2[1] = p2[1];
    t2[2] = p2[2];
    t2[3] = p2[3];

    t3[0] = pMid1[0];
    t3[1] = pMid1[1];
    t3[2] = pMid1[2];
    t3[3] = pMid1[3];

    t4[0] = pMid2[0];
    t4[1] = pMid2[1];
    t4[2] = pMid2[2];
    t4[3] = pMid2[3];

    *p1++ = t1[0] + t2[0];
    *p1++ = t1[1] + t2[1];
    *p1++ = t1[2] + t2[2];
    *p1++ = t1[3] + t2[3];    /* col 1 */

    t2[0] = t1[0] - t2[0];
    t2[1] = t1[1] - t2[1];
    t2[2] = t1[2] - t2[2];
    t2[3] = t1[3] - t2[3];    /* for col 2 */

    *pMid1++ = t3[0] + t4[0];
    *pMid1++ = t3[1] + t4[1];
    *pMid1++ = t3[2] + t4[2];
    *pMid1++ = t3[3] + t4[3]; /* col 1 */

    t4[0] = t4[0] - t3[0];
    t4[1] = t4[1] - t3[1];
    t4[2] = t4[2] - t3[2];
    t4[3] = t4[3] - t3[3];    /* for col 2 */

    twR = *tw++;
    twI = *tw++;

    /* multiply by twiddle factors */
    m0 = t2[0] * twR;
    m1 = t2[1] * twI;
    m2 = t2[1] * twR;
    m3 = t2[0] * twI;

    /* R  =  R  *  Tr - I * Ti */
    *p2++ = m0 + m1;
    /* I  =  I  *  Tr + R * Ti */
    *p2++ = m2 - m3;

    /* use vertical symmetry */
    /*  0.9988 - 0.0491i <==> -0.0491 - 0.9988i */
    m0 = t4[0] * twI;
    m1 = t4[1] * twR;
    m2 = t4[1] * twI;
    m3 = t4[0] * twR;

    *pMid2++ = m0 - m1;
    *pMid2++ = m2 + m3;

    twR = *tw++;
    twI = *tw++;

    m0 = t2[2] * twR;
    m1 = t2[3] * twI;
    m2 = t2[3] * twR;
    m3 = t2[2] * twI;

    *p2++ = m0 + m1;
    *p2++ = m2 - m3;

    m0 = t4[2] * twI;
    m1 = t4[3] * twR;
    m2 = t4[3] * twI;
    m3 = t4[2] * twR;

    *pMid2++ = m0 - m1;
    *pMid2++ = m2 + m3;
  }

  /* first col */
  arm_radix8_butterfly_f32 (pCol1, L, (float32_t *) S->pTwiddle, 2U);

  /* second col */
  arm_radix8_butterfly_f32 (pCol2, L, (float32_t *) S->pTwiddle, 2U);
}

/**
  brief         Core function for the floating-point CFFT butterfly process.
  param[in,out] pSrc             points to the in-place buffer of floating-point data type.
  param[in]     fftLen           length of the FFT.
  param[in]     pCoef            points to the twiddle coefficient buffer.
  param[in]     twidCoefModifier twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table.
  return        none
*/

void arm_radix8_butterfly_f32(
  float32_t * pSrc,
  uint16_t fftLen,
  const float32_t * pCoef,
  uint16_t twidCoefModifier)
{
   uint32_t ia1, ia2, ia3, ia4, ia5, ia6, ia7;
   uint32_t i1, i2, i3, i4, i5, i6, i7, i8;
   uint32_t id;
   uint32_t n1, n2, j;

   float32_t r1, r2, r3, r4, r5, r6, r7, r8;
   float32_t t1, t2;
   float32_t s1, s2, s3, s4, s5, s6, s7, s8;
   float32_t p1, p2, p3, p4;
   float32_t co2, co3, co4, co5, co6, co7, co8;
   float32_t si2, si3, si4, si5, si6, si7, si8;
   const float32_t C81 = 0.70710678118f;

   n2 = fftLen;

   do
   {
      n1 = n2;
      n2 = n2 >> 3;
      i1 = 0;

      do
      {
         i2 = i1 + n2;
         i3 = i2 + n2;
         i4 = i3 + n2;
         i5 = i4 + n2;
         i6 = i5 + n2;
         i7 = i6 + n2;
         i8 = i7 + n2;
         r1 = pSrc[2 * i1] + pSrc[2 * i5];
         r5 = pSrc[2 * i1] - pSrc[2 * i5];
         r2 = pSrc[2 * i2] + pSrc[2 * i6];
         r6 = pSrc[2 * i2] - pSrc[2 * i6];
         r3 = pSrc[2 * i3] + pSrc[2 * i7];
         r7 = pSrc[2 * i3] - pSrc[2 * i7];
         r4 = pSrc[2 * i4] + pSrc[2 * i8];
         r8 = pSrc[2 * i4] - pSrc[2 * i8];
         t1 = r1 - r3;
         r1 = r1 + r3;
         r3 = r2 - r4;
         r2 = r2 + r4;
         pSrc[2 * i1] = r1 + r2;
         pSrc[2 * i5] = r1 - r2;
         r1 = pSrc[2 * i1 + 1] + pSrc[2 * i5 + 1];
         s5 = pSrc[2 * i1 + 1] - pSrc[2 * i5 + 1];
         r2 = pSrc[2 * i2 + 1] + pSrc[2 * i6 + 1];
         s6 = pSrc[2 * i2 + 1] - pSrc[2 * i6 + 1];
         s3 = pSrc[2 * i3 + 1] + pSrc[2 * i7 + 1];
         s7 = pSrc[2 * i3 + 1] - pSrc[2 * i7 + 1];
         r4 = pSrc[2 * i4 + 1] + pSrc[2 * i8 + 1];
         s8 = pSrc[2 * i4 + 1] - pSrc[2 * i8 + 1];
         t2 = r1 - s3;
         r1 = r1 + s3;
         s3 = r2 - r4;
         r2 = r2 + r4;
         pSrc[2 * i1 + 1] = r1 + r2;
         pSrc[2 * i5 + 1] = r1 - r2;
         pSrc[2 * i3]     = t1 + s3;
         pSrc[2 * i7]     = t1 - s3;
         pSrc[2 * i3 + 1] = t2 - r3;
         pSrc[2 * i7 + 1] = t2 + r3;
         r1 = (r6 - r8) * C81;
         r6 = (r6 + r8) * C81;
         r2 = (s6 - s8) * C81;
         s6 = (s6 + s8) * C81;
         t1 = r5 - r1;
         r5 = r5 + r1;
         r8 = r7 - r6;
         r7 = r7 + r6;
         t2 = s5 - r2;
         s5 = s5 + r2;
         s8 = s7 - s6;
         s7 = s7 + s6;
         pSrc[2 * i2]     = r5 + s7;
         pSrc[2 * i8]     = r5 - s7;
         pSrc[2 * i6]     = t1 + s8;
         pSrc[2 * i4]     = t1 - s8;
         pSrc[2 * i2 + 1] = s5 - r7;
         pSrc[2 * i8 + 1] = s5 + r7;
         pSrc[2 * i6 + 1] = t2 - r8;
         pSrc[2 * i4 + 1] = t2 + r8;

         i1 += n1;
      } while (i1 < fftLen);

      if (n2 < 8)
         break;

      ia1 = 0;
      j = 1;

      do
      {
         /*  index calculation for the coefficients */
         id  = ia1 + twidCoefModifier;
         ia1 = id;
         ia2 = ia1 + id;
         ia3 = ia2 + id;
         ia4 = ia3 + id;
         ia5 = ia4 + id;
         ia6 = ia5 + id;
         ia7 = ia6 + id;

         co2 = pCoef[2 * ia1];
         co3 = pCoef[2 * ia2];
         co4 = pCoef[2 * ia3];
         co5 = pCoef[2 * ia4];
         co6 = pCoef[2 * ia5];
         co7 = pCoef[2 * ia6];
         co8 = pCoef[2 * ia7];
         si2 = pCoef[2 * ia1 + 1];
         si3 = pCoef[2 * ia2 + 1];
         si4 = pCoef[2 * ia3 + 1];
         si5 = pCoef[2 * ia4 + 1];
         si6 = pCoef[2 * ia5 + 1];
         si7 = pCoef[2 * ia6 + 1];
         si8 = pCoef[2 * ia7 + 1];

         i1 = j;

         do
         {
            /*  index calculation for the input */
            i2 = i1 + n2;
            i3 = i2 + n2;
            i4 = i3 + n2;
            i5 = i4 + n2;
            i6 = i5 + n2;
            i7 = i6 + n2;
            i8 = i7 + n2;
            r1 = pSrc[2 * i1] + pSrc[2 * i5];
            r5 = pSrc[2 * i1] - pSrc[2 * i5];
            r2 = pSrc[2 * i2] + pSrc[2 * i6];
            r6 = pSrc[2 * i2] - pSrc[2 * i6];
            r3 = pSrc[2 * i3] + pSrc[2 * i7];
            r7 = pSrc[2 * i3] - pSrc[2 * i7];
            r4 = pSrc[2 * i4] + pSrc[2 * i8];
            r8 = pSrc[2 * i4] - pSrc[2 * i8];
            t1 = r1 - r3;
            r1 = r1 + r3;
            r3 = r2 - r4;
            r2 = r2 + r4;
            pSrc[2 * i1] = r1 + r2;
            r2 = r1 - r2;
            s1 = pSrc[2 * i1 + 1] + pSrc[2 * i5 + 1];
            s5 = pSrc[2 * i1 + 1] - pSrc[2 * i5 + 1];
            s2 = pSrc[2 * i2 + 1] + pSrc[2 * i6 + 1];
            s6 = pSrc[2 * i2 + 1] - pSrc[2 * i6 + 1];
            s3 = pSrc[2 * i3 + 1] + pSrc[2 * i7 + 1];
            s7 = pSrc[2 * i3 + 1] - pSrc[2 * i7 + 1];
            s4 = pSrc[2 * i4 + 1] + pSrc[2 * i8 + 1];
            s8 = pSrc[2 * i4 + 1] - pSrc[2 * i8 + 1];
            t2 = s1 - s3;
            s1 = s1 + s3;
            s3 = s2 - s4;
            s2 = s2 + s4;
            r1 = t1 + s3;
            t1 = t1 - s3;
            pSrc[2 * i1 + 1] = s1 + s2;
            s2 = s1 - s2;
            s1 = t2 - r3;
            t2 = t2 + r3;
            p1 = co5 * r2;
            p2 = si5 * s2;
            p3 = co5 * s2;
            p4 = si5 * r2;
            pSrc[2 * i5]     = p1 + p2;
            pSrc[2 * i5 + 1] = p3 - p4;
            p1 = co3 * r1;
            p2 = si3 * s1;
            p3 = co3 * s1;
            p4 = si3 * r1;
            pSrc[2 * i3]     = p1 + p2;
            pSrc[2 * i3 + 1] = p3 - p4;
            p1 = co7 * t1;
            p2 = si7 * t2;
            p3 = co7 * t2;
            p4 = si7 * t1;
            pSrc[2 * i7]     = p1 + p2;
            pSrc[2 * i7 + 1] = p3 - p4;
            r1 = (r6 - r8) * C81;
            r6 = (r6 + r8) * C81;
            s1 = (s6 - s8) * C81;
            s6 = (s6 + s8) * C81;
            t1 = r5 - r1;
            r5 = r5 + r1;
            r8 = r7 - r6;
            r7 = r7 + r6;
            t2 = s5 - s1;
            s5 = s5 + s1;
            s8 = s7 - s6;
            s7 = s7 + s6;
            r1 = r5 + s7;
            r5 = r5 - s7;
            r6 = t1 + s8;
            t1 = t1 - s8;
            s1 = s5 - r7;
            s5 = s5 + r7;
            s6 = t2 - r8;
            t2 = t2 + r8;
            p1 = co2 * r1;
            p2 = si2 * s1;
            p3 = co2 * s1;
            p4 = si2 * r1;
            pSrc[2 * i2]     = p1 + p2;
            pSrc[2 * i2 + 1] = p3 - p4;
            p1 = co8 * r5;
            p2 = si8 * s5;
            p3 = co8 * s5;
            p4 = si8 * r5;
            pSrc[2 * i8]     = p1 + p2;
            pSrc[2 * i8 + 1] = p3 - p4;
            p1 = co6 * r6;
            p2 = si6 * s6;
            p3 = co6 * s6;
            p4 = si6 * r6;
            pSrc[2 * i6]     = p1 + p2;
            pSrc[2 * i6 + 1] = p3 - p4;
            p1 = co4 * t1;
            p2 = si4 * t2;
            p3 = co4 * t2;
            p4 = si4 * t1;
            pSrc[2 * i4]     = p1 + p2;
            pSrc[2 * i4 + 1] = p3 - p4;

            i1 += n1;
         } while (i1 < fftLen);

         j++;
      } while (j < n2);

      twidCoefModifier <<= 3;
   } while (n2 > 7);
}

/**
  @brief         Processing function for the floating-point complex FFT.
  @param[in]     S              points to an instance of the floating-point CFFT structure
  @param[in,out] p1             points to the complex data buffer of size <code>2*fftLen</code>. Processing occurs in-place
  @param[in]     ifftFlag       flag that selects transform direction
                   - value = 0: forward transform
                   - value = 1: inverse transform
  @param[in]     bitReverseFlag flag that enables / disables bit reversal of output
                   - value = 0: disables bit reversal of output
                   - value = 1: enables bit reversal of output
  @return        none
 */

void arm_cfft_radix8by4_f32 (arm_cfft_instance_f32 * S, float32_t * p1)
{
    uint32_t    L  = S->fftLen >> 1;
    float32_t * pCol1, *pCol2, *pCol3, *pCol4, *pEnd1, *pEnd2, *pEnd3, *pEnd4;
    const float32_t *tw2, *tw3, *tw4;
    float32_t * p2 = p1 + L;
    float32_t * p3 = p2 + L;
    float32_t * p4 = p3 + L;
    float32_t t2[4], t3[4], t4[4], twR, twI;
    float32_t p1ap3_0, p1sp3_0, p1ap3_1, p1sp3_1;
    float32_t m0, m1, m2, m3;
    uint32_t l, twMod2, twMod3, twMod4;

    pCol1 = p1;         /* points to real values by default */
    pCol2 = p2;
    pCol3 = p3;
    pCol4 = p4;
    pEnd1 = p2 - 1;     /* points to imaginary values by default */
    pEnd2 = p3 - 1;
    pEnd3 = p4 - 1;
    pEnd4 = pEnd3 + L;

    tw2 = tw3 = tw4 = (float32_t *) S->pTwiddle;

    L >>= 1;

    /* do four dot Fourier transform */

    twMod2 = 2;
    twMod3 = 4;
    twMod4 = 6;

    /* TOP */
    p1ap3_0 = p1[0] + p3[0];
    p1sp3_0 = p1[0] - p3[0];
    p1ap3_1 = p1[1] + p3[1];
    p1sp3_1 = p1[1] - p3[1];

    /* col 2 */
    t2[0] = p1sp3_0 + p2[1] - p4[1];
    t2[1] = p1sp3_1 - p2[0] + p4[0];
    /* col 3 */
    t3[0] = p1ap3_0 - p2[0] - p4[0];
    t3[1] = p1ap3_1 - p2[1] - p4[1];
    /* col 4 */
    t4[0] = p1sp3_0 - p2[1] + p4[1];
    t4[1] = p1sp3_1 + p2[0] - p4[0];
    /* col 1 */
    *p1++ = p1ap3_0 + p2[0] + p4[0];
    *p1++ = p1ap3_1 + p2[1] + p4[1];

    /* Twiddle factors are ones */
    *p2++ = t2[0];
    *p2++ = t2[1];
    *p3++ = t3[0];
    *p3++ = t3[1];
    *p4++ = t4[0];
    *p4++ = t4[1];

    tw2 += twMod2;
    tw3 += twMod3;
    tw4 += twMod4;

    for (l = (L - 2) >> 1; l > 0; l-- )
    {
      /* TOP */
      p1ap3_0 = p1[0] + p3[0];
      p1sp3_0 = p1[0] - p3[0];
      p1ap3_1 = p1[1] + p3[1];
      p1sp3_1 = p1[1] - p3[1];
      /* col 2 */
      t2[0] = p1sp3_0 + p2[1] - p4[1];
      t2[1] = p1sp3_1 - p2[0] + p4[0];
      /* col 3 */
      t3[0] = p1ap3_0 - p2[0] - p4[0];
      t3[1] = p1ap3_1 - p2[1] - p4[1];
      /* col 4 */
      t4[0] = p1sp3_0 - p2[1] + p4[1];
      t4[1] = p1sp3_1 + p2[0] - p4[0];
      /* col 1 - top */
      *p1++ = p1ap3_0 + p2[0] + p4[0];
      *p1++ = p1ap3_1 + p2[1] + p4[1];

      /* BOTTOM */
      p1ap3_1 = pEnd1[-1] + pEnd3[-1];
      p1sp3_1 = pEnd1[-1] - pEnd3[-1];
      p1ap3_0 = pEnd1[ 0] + pEnd3[0];
      p1sp3_0 = pEnd1[ 0] - pEnd3[0];
      /* col 2 */
      t2[2] = pEnd2[0] - pEnd4[0] + p1sp3_1;
      t2[3] = pEnd1[0] - pEnd3[0] - pEnd2[-1] + pEnd4[-1];
      /* col 3 */
      t3[2] = p1ap3_1 - pEnd2[-1] - pEnd4[-1];
      t3[3] = p1ap3_0 - pEnd2[ 0] - pEnd4[ 0];
      /* col 4 */
      t4[2] = pEnd2[ 0] - pEnd4[ 0] - p1sp3_1;
      t4[3] = pEnd4[-1] - pEnd2[-1] - p1sp3_0;
      /* col 1 - Bottom */
      *pEnd1-- = p1ap3_0 + pEnd2[ 0] + pEnd4[ 0];
      *pEnd1-- = p1ap3_1 + pEnd2[-1] + pEnd4[-1];

      /* COL 2 */
      /* read twiddle factors */
      twR = *tw2++;
      twI = *tw2++;
      /* multiply by twiddle factors */
      /*  let    Z1 = a + i(b),   Z2 = c + i(d) */
      /*   =>  Z1 * Z2  =  (a*c - b*d) + i(b*c + a*d) */

      /* Top */
      m0 = t2[0] * twR;
      m1 = t2[1] * twI;
      m2 = t2[1] * twR;
      m3 = t2[0] * twI;

      *p2++ = m0 + m1;
      *p2++ = m2 - m3;
      /* use vertical symmetry col 2 */
      /* 0.9997 - 0.0245i  <==>  0.0245 - 0.9997i */
      /* Bottom */
      m0 = t2[3] * twI;
      m1 = t2[2] * twR;
      m2 = t2[2] * twI;
      m3 = t2[3] * twR;

      *pEnd2-- = m0 - m1;
      *pEnd2-- = m2 + m3;

      /* COL 3 */
      twR = tw3[0];
      twI = tw3[1];
      tw3 += twMod3;
      /* Top */
      m0 = t3[0] * twR;
      m1 = t3[1] * twI;
      m2 = t3[1] * twR;
      m3 = t3[0] * twI;

      *p3++ = m0 + m1;
      *p3++ = m2 - m3;
      /* use vertical symmetry col 3 */
      /* 0.9988 - 0.0491i  <==>  -0.9988 - 0.0491i */
      /* Bottom */
      m0 = -t3[3] * twR;
      m1 =  t3[2] * twI;
      m2 =  t3[2] * twR;
      m3 =  t3[3] * twI;

      *pEnd3-- = m0 - m1;
      *pEnd3-- = m3 - m2;

      /* COL 4 */
      twR = tw4[0];
      twI = tw4[1];
      tw4 += twMod4;
      /* Top */
      m0 = t4[0] * twR;
      m1 = t4[1] * twI;
      m2 = t4[1] * twR;
      m3 = t4[0] * twI;

      *p4++ = m0 + m1;
      *p4++ = m2 - m3;
      /* use vertical symmetry col 4 */
      /* 0.9973 - 0.0736i  <==>  -0.0736 + 0.9973i */
      /* Bottom */
      m0 = t4[3] * twI;
      m1 = t4[2] * twR;
      m2 = t4[2] * twI;
      m3 = t4[3] * twR;

      *pEnd4-- = m0 - m1;
      *pEnd4-- = m2 + m3;
    }

    /* MIDDLE */
    /* Twiddle factors are */
    /*  1.0000  0.7071-0.7071i  -1.0000i  -0.7071-0.7071i */
    p1ap3_0 = p1[0] + p3[0];
    p1sp3_0 = p1[0] - p3[0];
    p1ap3_1 = p1[1] + p3[1];
    p1sp3_1 = p1[1] - p3[1];

    /* col 2 */
    t2[0] = p1sp3_0 + p2[1] - p4[1];
    t2[1] = p1sp3_1 - p2[0] + p4[0];
    /* col 3 */
    t3[0] = p1ap3_0 - p2[0] - p4[0];
    t3[1] = p1ap3_1 - p2[1] - p4[1];
    /* col 4 */
    t4[0] = p1sp3_0 - p2[1] + p4[1];
    t4[1] = p1sp3_1 + p2[0] - p4[0];
    /* col 1 - Top */
    *p1++ = p1ap3_0 + p2[0] + p4[0];
    *p1++ = p1ap3_1 + p2[1] + p4[1];

    /* COL 2 */
    twR = tw2[0];
    twI = tw2[1];

    m0 = t2[0] * twR;
    m1 = t2[1] * twI;
    m2 = t2[1] * twR;
    m3 = t2[0] * twI;

    *p2++ = m0 + m1;
    *p2++ = m2 - m3;
    /* COL 3 */
    twR = tw3[0];
    twI = tw3[1];

    m0 = t3[0] * twR;
    m1 = t3[1] * twI;
    m2 = t3[1] * twR;
    m3 = t3[0] * twI;

    *p3++ = m0 + m1;
    *p3++ = m2 - m3;
    /* COL 4 */
    twR = tw4[0];
    twI = tw4[1];

    m0 = t4[0] * twR;
    m1 = t4[1] * twI;
    m2 = t4[1] * twR;
    m3 = t4[0] * twI;

    *p4++ = m0 + m1;
    *p4++ = m2 - m3;

    /* first col */
    arm_radix8_butterfly_f32 (pCol1, L, (float32_t *) S->pTwiddle, 4U);

    /* second col */
    arm_radix8_butterfly_f32 (pCol2, L, (float32_t *) S->pTwiddle, 4U);

    /* third col */
    arm_radix8_butterfly_f32 (pCol3, L, (float32_t *) S->pTwiddle, 4U);

    /* fourth col */
    arm_radix8_butterfly_f32 (pCol4, L, (float32_t *) S->pTwiddle, 4U);
}

void arm_cfft_f32(
  const arm_cfft_instance_f32 * S,
        float32_t * p1,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag)
{
  uint32_t  L = S->fftLen, l;
  float32_t invL, * pSrc;

  if (ifftFlag == 1U)
  {
    /* Conjugate input data */
    pSrc = p1 + 1;
    for (l = 0; l < L; l++)
    {
      *pSrc = -*pSrc;
      pSrc += 2;
    }
  }

  switch (L)
  {
  case 16:
  case 128:
  case 1024:
    arm_cfft_radix8by2_f32 ( (arm_cfft_instance_f32 *) S, p1);
    break;
  case 32:
  case 256:
  case 2048:
    arm_cfft_radix8by4_f32 ( (arm_cfft_instance_f32 *) S, p1);
    break;
  case 64:
  case 512:
  case 4096:
    arm_radix8_butterfly_f32 ( p1, L, (float32_t *) S->pTwiddle, 1);
    break;
  }

  if ( bitReverseFlag )
    arm_bitreversal_32 ((uint32_t*) p1, S->bitRevLength, S->pBitRevTable);

  if (ifftFlag == 1U)
  {
    invL = 1.0f / (float32_t)L;

    /* Conjugate and scale output data */
    pSrc = p1;
    for (l= 0; l < L; l++)
    {
      *pSrc++ *=   invL ;
      *pSrc    = -(*pSrc) * invL;
      pSrc++;
    }
  }
}

/**
  @brief         In-place 32 bit reversal function.
  @param[in,out] pSrc        points to in-place buffer of unknown 32-bit data type
  @param[in]     bitRevLen   bit reversal table length
  @param[in]     pBitRevTab  points to bit reversal table
  @return        none
*/

void arm_bitreversal_32(
        uint32_t *pSrc,
  const uint16_t bitRevLen,
  const uint16_t *pBitRevTab)
{
  uint32_t a, b, i, tmp;

  for (i = 0; i < bitRevLen; )
  {
     a = pBitRevTab[i    ] >> 2;
     b = pBitRevTab[i + 1] >> 2;

     //real
     tmp = pSrc[a];
     pSrc[a] = pSrc[b];
     pSrc[b] = tmp;

     //complex
     tmp = pSrc[a+1];
     pSrc[a+1] = pSrc[b+1];
     pSrc[b+1] = tmp;

    i += 2;
  }
}

/* Prepares data for inverse cfft */
void merge_rfft_f32(
  const arm_rfft_fast_instance_f32 * S,
        float32_t * p,
        float32_t * pOut)
{
        int32_t  k;                                /* Loop Counter */
        float32_t twR, twI;                         /* RFFT Twiddle coefficients */
  const float32_t *pCoeff = S->pTwiddleRFFT;        /* Points to RFFT Twiddle factors */
        float32_t *pA = p;                          /* increasing pointer */
        float32_t *pB = p;                          /* decreasing pointer */
        float32_t xAR, xAI, xBR, xBI;               /* temporary variables */
        float32_t t1a, t1b, r, s, t, u;             /* temporary variables */

   k = (S->Sint).fftLen - 1;

   xAR = pA[0];
   xAI = pA[1];

   pCoeff += 2 ;

   *pOut++ = 0.5f * ( xAR + xAI );
   *pOut++ = 0.5f * ( xAR - xAI );

   pB  =  p + 2*k ;
   pA +=  2	   ;

   while (k > 0)
   {
      /* G is half of the frequency complex spectrum */
      //for k = 2:N
      //    Xk(k) = 1/2 * (G(k) + conj(G(N-k+2)) + Tw(k)*( G(k) - conj(G(N-k+2))));
      xBI =   pB[1]    ;
      xBR =   pB[0]    ;
      xAR =  pA[0];
      xAI =  pA[1];

      twR = *pCoeff++;
      twI = *pCoeff++;

      t1a = xAR - xBR ;
      t1b = xAI + xBI ;

      r = twR * t1a;
      s = twI * t1b;
      t = twI * t1a;
      u = twR * t1b;

      // real(tw * (xA - xB)) = twR * (xAR - xBR) - twI * (xAI - xBI);
      // imag(tw * (xA - xB)) = twI * (xAR - xBR) + twR * (xAI - xBI);
      *pOut++ = 0.5f * (xAR + xBR - r - s ); //xAR
      *pOut++ = 0.5f * (xAI - xBI + t - u ); //xAI

      pA += 2;
      pB -= 2;
      k--;
   }
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

void arm_rms_f32(
  const float32_t * pSrc,
        uint32_t blockSize,
        float32_t * pResult)
{
        uint32_t blkCnt;                               /* Loop counter */
        float32_t sum = 0.0f;                          /* Temporary result storage */
        float32_t in;                                  /* Temporary variable to store input value */

  
  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

  while (blkCnt > 0U)
  {
    /* C = A[0] * A[0] + A[1] * A[1] + ... + A[blockSize-1] * A[blockSize-1] */

    in = *pSrc++;
    /* Compute sum of squares and store result in a temporary variable. */
    sum += ( in * in);

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Compute Rms and store result in destination */
  arm_sqrt_f32(sum / (float32_t) blockSize, pResult);
}

/**
  @brief         Floating-point square root function.
  @param[in]     in    input value
  @param[out]    pOut  square root of input value
  @return        execution status
                   - \ref ARM_MATH_SUCCESS        : input value is positive
                   - \ref ARM_MATH_ARGUMENT_ERROR : input value is negative; *pOut is set to 0
 */
__STATIC_FORCEINLINE int arm_sqrt_f32(
  float32_t in,
  float32_t * pOut)
  {
    if (in >= 0.0f)
    {
      *pOut = sqrtf(in);
      return 0;
    }
    else
    {
      *pOut = 0.0f;
      return -1;
    }
  }


// ---------------------------------------------- NUMPY DEPENDENCIES (ARM FFT) END ----------------------------------------------
// ---------------------------------------------- NUMPY DEPENDENCIES (ARM FFT) END ----------------------------------------------



// ---------------------------------------------- NUMPY DEPENDENCIES (KISSFFT) START ----------------------------------------------
// ---------------------------------------------- NUMPY DEPENDENCIES (KISSFFT) START ----------------------------------------------

#define kiss_fft_scalar float
#define KISS_FFT_COS(phase) (kiss_fft_scalar) cos(phase)
#define KISS_FFT_SIN(phase) (kiss_fft_scalar) sin(phase)

#define kf_cexp(x,phase) \
	do{ \
		(x)->r = KISS_FFT_COS(phase);\
		(x)->i = KISS_FFT_SIN(phase);\
	}while(0)

typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
}kiss_fft_cpx;

typedef struct kiss_fft_state* kiss_fft_cfg;

typedef struct kiss_fftr_state *kiss_fftr_cfg;

struct kiss_fft_state{
    int nfft;
    int inverse;
    int factors[2*32];
    kiss_fft_cpx twiddles[1];
};

struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
};

/*  facbuf is populated by p1,m1,p2,m2, ...
    where
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static
void kf_factor(int n,int * facbuf)
{
    int p=4;
    double floor_sqrt;
    floor_sqrt = floor(sqrt((double)n));

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}

/*
 *
 * User-callable function to allocate all necessary storage space for the fft.
 *
 * The return value is a contiguous block of memory, allocated with malloc.  As such,
 * It can be freed with free(), rather than a kiss_fft-specific function.
 * */
kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem,size_t * memallocated=NULL)
{
    kiss_fft_cfg st=NULL;
    size_t memneeded = sizeof(struct kiss_fft_state)
        + sizeof(kiss_fft_cpx)*(nfft-1); /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( kiss_fft_cfg)malloc(memneeded);
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        int i;
        st->nfft=nfft;
        st->inverse = inverse_fft;
        if (inverse_fft)
        {
            for (i=0;i<nfft;++i) {
                const double pi=3.141592653589793238462643383279502884197169399375105820974944;
                double phase = 2*pi*i / nfft;
                kf_cexp(st->twiddles+i, phase );
            }
        } else {
            for (i=0;i<nfft;++i) {
                const double pi=3.141592653589793238462643383279502884197169399375105820974944;
                double phase = -2*pi*i / nfft;
                kf_cexp(st->twiddles+i, phase );
            }
        }

        kf_factor(nfft,st->factors);
    }

    if (memallocated != NULL) {
        *memallocated = memneeded;
    }

    return st;
}

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem,size_t * memallocated=NULL)
{
    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize = 0, memneeded;

    if (nfft & 1) {
        serial_printf("FFT length must be even\n");
        return NULL;
    }
    nfft >>= 1;

    kiss_fft_alloc(nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) + subsize + sizeof(kiss_fft_cpx) * ( nfft * 3 / 2);

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg) malloc(memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx *) (((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

    if (inverse_fft) {
        for (i = 0; i < nfft/2; ++i) {
            double phase =
                3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
            kf_cexp (st->super_twiddles+i,phase);
        }
    } else  {
        for (i = 0; i < nfft/2; ++i) {
            double phase =
                -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);
            kf_cexp (st->super_twiddles+i,phase);
        }
    }

    if (memallocated != NULL) {
        *memallocated = memneeded;
    }

    return st;
}

static
void kf_work(
        kiss_fft_cpx * Fout,
        const kiss_fft_cpx * f,
        const size_t fstride,
        int in_stride,
        int * factors,
        const kiss_fft_cfg st
        )
{
    kiss_fft_cpx * Fout_beg=Fout;
    const int p=*factors++; /* the radix  */
    const int m=*factors++; /* stage's fft length/p */
    const kiss_fft_cpx * Fout_end = Fout + p*m;

    if (m==1) {
        do{
            *Fout = *f;
            f += fstride*in_stride;
        }while(++Fout != Fout_end );
    }else{
        do{
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m,
            // each one takes a decimated version of the input
            kf_work( Fout , f, fstride*p, in_stride, factors,st);
            f += fstride*in_stride;
        }while( (Fout += m) != Fout_end );
    }

    Fout=Fout_beg;

    // recombine the p smaller DFTs
    switch (p) {
        case 2: kf_bfly2(Fout,fstride,st,m); break;
        case 3: kf_bfly3(Fout,fstride,st,m); break;
        case 4: kf_bfly4(Fout,fstride,st,m); break;
        case 5: kf_bfly5(Fout,fstride,st,m); break;
        default: kf_bfly_generic(Fout,fstride,st,m,p); break;
    }
}

static void kf_bfly5(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
    int u;
    kiss_fft_cpx scratch[13];
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx *tw;
    kiss_fft_cpx ya,yb;
    ya = twiddles[fstride*m];
    yb = twiddles[fstride*2*m];

    Fout0=Fout;
    Fout1=Fout0+m;
    Fout2=Fout0+2*m;
    Fout3=Fout0+3*m;
    Fout4=Fout0+4*m;

    tw=st->twiddles;
    for ( u=0; u<m; ++u ) {
        C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
        scratch[0] = *Fout0;

        C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
        C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
        C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
        C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

        C_ADD( scratch[7],scratch[1],scratch[4]);
        C_SUB( scratch[10],scratch[1],scratch[4]);
        C_ADD( scratch[8],scratch[2],scratch[3]);
        C_SUB( scratch[9],scratch[2],scratch[3]);

        Fout0->r += scratch[7].r + scratch[8].r;
        Fout0->i += scratch[7].i + scratch[8].i;

        scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
        scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

        scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
        scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

        C_SUB(*Fout1,scratch[5],scratch[6]);
        C_ADD(*Fout4,scratch[5],scratch[6]);

        scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
        scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
        scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
        scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

        C_ADD(*Fout2,scratch[11],scratch[12]);
        C_SUB(*Fout3,scratch[11],scratch[12]);

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
    }
}

/* perform the butterfly for one stage of a mixed radix FFT */
static void kf_bfly_generic(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m,
        int p
        )
{
    int u,k,q1,q;
    kiss_fft_cpx * twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_cpx * scratch = (kiss_fft_cpx*)KISS_FFT_TMP_ALLOC(sizeof(kiss_fft_cpx)*p);

    for ( u=0; u<m; ++u ) {
        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            scratch[q1] = Fout[ k  ];
            C_FIXDIV(scratch[q1],p);
            k += m;
        }

        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
            int twidx=0;
            Fout[ k ] = scratch[0];
            for (q=1;q<p;++q ) {
                twidx += fstride * k;
                if (twidx>=Norig) twidx-=Norig;
                C_MUL(t,scratch[q] , twiddles[twidx] );
                C_ADDTO( Fout[ k ] ,t);
            }
            k += m;
        }
    }
    KISS_FFT_TMP_FREE(scratch);
}

/* The guts header contains all the multiplication and addition macros that are defined for
 fixed or floating point complex numbers.  It also delares the kf_ internal functions.
 */

static void kf_bfly2(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        int m
        )
{
    kiss_fft_cpx * Fout2;
    kiss_fft_cpx * tw1 = st->twiddles;
    kiss_fft_cpx t;
    Fout2 = Fout + m;
    do{
        C_FIXDIV(*Fout,2); C_FIXDIV(*Fout2,2);

        C_MUL (t,  *Fout2 , *tw1);
        tw1 += fstride;
        C_SUB( *Fout2 ,  *Fout , t );
        C_ADDTO( *Fout ,  t );
        ++Fout2;
        ++Fout;
    }while (--m);
}

static void kf_bfly4(
        kiss_fft_cpx * Fout,
        const size_t fstride,
        const kiss_fft_cfg st,
        const size_t m
        )
{
    kiss_fft_cpx *tw1,*tw2,*tw3;
    kiss_fft_cpx scratch[6];
    size_t k=m;
    const size_t m2=2*m;
    const size_t m3=3*m;


    tw3 = tw2 = tw1 = st->twiddles;

    do {
        C_FIXDIV(*Fout,4); C_FIXDIV(Fout[m],4); C_FIXDIV(Fout[m2],4); C_FIXDIV(Fout[m3],4);

        C_MUL(scratch[0],Fout[m] , *tw1 );
        C_MUL(scratch[1],Fout[m2] , *tw2 );
        C_MUL(scratch[2],Fout[m3] , *tw3 );

        C_SUB( scratch[5] , *Fout, scratch[1] );
        C_ADDTO(*Fout, scratch[1]);
        C_ADD( scratch[3] , scratch[0] , scratch[2] );
        C_SUB( scratch[4] , scratch[0] , scratch[2] );
        C_SUB( Fout[m2], *Fout, scratch[3] );
        tw1 += fstride;
        tw2 += fstride*2;
        tw3 += fstride*3;
        C_ADDTO( *Fout , scratch[3] );

        if(st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        }else{
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    }while(--k);
}

static void kf_bfly3(
         kiss_fft_cpx * Fout,
         const size_t fstride,
         const kiss_fft_cfg st,
         size_t m
         )
{
     size_t k=m;
     const size_t m2 = 2*m;
     kiss_fft_cpx *tw1,*tw2;
     kiss_fft_cpx scratch[5];
     kiss_fft_cpx epi3;
     epi3 = st->twiddles[fstride*m];

     tw1=tw2=st->twiddles;

     do{
         C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

         C_MUL(scratch[1],Fout[m] , *tw1);
         C_MUL(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
     }while(--k);
}



void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    kiss_fft_stride(cfg,fin,fout,1);
}

void kiss_fft_stride(kiss_fft_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int in_stride)
{
    if (fin == fout) {
        //NOTE: this is not really an in-place FFT algorithm.
        //It just performs an out-of-place FFT into a temp buffer
        kiss_fft_cpx * tmpbuf = (kiss_fft_cpx*)malloc( sizeof(kiss_fft_cpx)*st->nfft);
        kf_work(tmpbuf,fin,1,in_stride, st->factors,st);
        memcpy(fout,tmpbuf,sizeof(kiss_fft_cpx)*st->nfft);
        KISS_FFT_TMP_FREE(tmpbuf);
    }else{
        kf_work( fout, fin, 1,in_stride, st->factors,st );
    }
}

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    kiss_fft_cpx fpnk,fpk,f1k,f2k,tw,tdc;

    if ( st->substate->inverse) {
        serial_printf("kiss fft usage error: improper alloc\n");
    }

    ncfft = st->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence.
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
     *      yielding Nyquist bin of input time sequence
     */

    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;

    freqdata[ncfft].i = freqdata[0].i = 0;

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k];
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k-1]);

        freqdata[k].r = HALF_OF(f1k.r + tw.r);
        freqdata[k].i = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = HALF_OF(tw.i - f1k.i);
    }
}

// ---------------------------------------------- NUMPY DEPENDENCIES (KISSFFT) END ----------------------------------------------
// ---------------------------------------------- NUMPY DEPENDENCIES (KISSFFT) END ----------------------------------------------

// !!---------------------------------------------- NUMPY CLASS START ----------------------------------------------!!
// !!---------------------------------------------- NUMPY CLASS START ----------------------------------------------!!

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

    static float sum(float *input_array, size_t input_array_size) {
        float res = 0.0f;
        for (size_t ix = 0; ix < input_array_size; ix++) {
            res += input_array[ix];
        }
        return res;
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
     * Get the minimum value in a matrix per row
     * @param input_matrix Input matrix (MxN)
     * @param output_matrix Output matrix (Mx1)
     */
    static int min(matrix_t *input_matrix, matrix_t *output_matrix) {
        if (input_matrix->rows != output_matrix->rows) {
            return -1;
        }
        if (output_matrix->cols != 1) {
            return -1;
        }

        for (size_t row = 0; row < input_matrix->rows; row++) {


            float min = FLT_MAX;

            for (size_t col = 0; col < input_matrix->cols; col++) {
                float v = input_matrix->buffer[( row * input_matrix->cols ) + col];
                if (v < min) {
                    min = v;
                }
            }

            output_matrix->buffer[row] = min;

        }

        return 0;
    }

    /**
     * > 50% faster then the math.h log() function
     * in return for a small loss in accuracy (0.00001 average diff with log())
     * From: https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c/39822314#39822314
     * Licensed under the CC BY-SA 3.0
     * @param a Input number
     * @returns Natural log value of a
     */
    __attribute__((always_inline)) static inline float log(float a)
    {
        float m, r, s, t, i, f;
        int32_t e, g;

        g = (int32_t) * ((int32_t *)&a);
        e = (g - 0x3f2aaaab) & 0xff800000;
        g = g - e;
        m = (float) * ((float *)&g);
        i = (float)e * 1.19209290e-7f; // 0x1.0p-23
        /* m in [2/3, 4/3] */
        f = m - 1.0f;
        s = f * f;
        /* Compute log1p(f) for f in [-1/3, 1/3] */
        r = fmaf(0.230836749f, f, -0.279208571f); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        t = fmaf(0.331826031f, f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2
        r = fmaf(r, s, t);
        r = fmaf(r, s, f);
        r = fmaf(i, 0.693147182f, r); // 0x1.62e430p-1 // log(2)

        return r;
    }

    /**
     * Calculate the natural log value of a matrix. Does an in-place replacement.
     * @param matrix Matrix (MxN)
     * @returns 0 if OK
     */
    static int log(matrix_t *matrix)
    {
        for (uint32_t ix = 0; ix < matrix->rows * matrix->cols; ix++) {
            matrix->buffer[ix] = numpy::log(matrix->buffer[ix]);
        }

        return 0;
    }

    /**
     * Get the maximum value in a matrix per row
     * @param input_matrix Input matrix (MxN)
     * @param output_matrix Output matrix (Mx1)
     */
    static int max(matrix_t *input_matrix, matrix_t *output_matrix) {
        if (input_matrix->rows != output_matrix->rows) {
            return -1;
        }
        if (output_matrix->cols != 1) {
            return -1;
        }

        for (size_t row = 0; row < input_matrix->rows; row++) {
            float max = -FLT_MAX;

            for (size_t col = 0; col < input_matrix->cols; col++) {
                float v = input_matrix->buffer[( row * input_matrix->cols ) + col];
                if (v > max) {
                    max = v;
                }
            }

            output_matrix->buffer[row] = max;
        }

        return 0;
    }

    /**
     * Multiply two matrices lazily per row in matrix 1 (MxN * NxK matrix)
     * @param i matrix1 row index
     * @param row matrix1 row
     * @param matrix1_cols matrix1 row size (1xN)
     * @param matrix2 Pointer to matrix2 (NxK)
     * @param out_matrix Pointer to out matrix (MxK)
     * @returns EIDSP_OK if OK
     */
    static inline int dot_by_row(int i, float *row, uint32_t matrix1_cols, matrix_t *matrix2, matrix_t *out_matrix) {
        if (matrix1_cols != matrix2->rows) {
            return -1;
        }

        for (size_t j = 0; j < matrix2->cols; j++) {
            float tmp = 0.0f;
            for (size_t k = 0; k < matrix1_cols; k++) {
                tmp += row[k] * matrix2->buffer[k * matrix2->cols + j];
            }
            out_matrix->buffer[i * matrix2->cols + j] += tmp;
        }

        return 0;
    }

    /**
     * Subtract from matrix in place
     * @param matrix
     * @param subtraction
     * @returns 0 if OK
     */
    static int subtract(matrix_t *matrix, float subtraction) {
        for (uint32_t ix = 0; ix < matrix->rows * matrix->cols; ix++) {
            matrix->buffer[ix] -= subtraction;
        }
        return 0;
    }

    /**
     * Normalize a matrix to 0..1. Does an in-place replacement.
     * Normalization done per row.
     * @param matrix
     */
    static int normalize(matrix_t *matrix) {
        // Python implementation:
        //  matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        int r;

        matrix_t temp_matrix(1, matrix->rows * matrix->cols, matrix->buffer);

        matrix_t min_matrix(1, 1);
        if (!min_matrix.buffer) {
            return -1;
        }
        r = min(&temp_matrix, &min_matrix);
        if (r != 0) {
            return -1;
        }

        matrix_t max_matrix(1, 1);
        if (!max_matrix.buffer) {
            return -1;
        }
        r = max(&temp_matrix, &max_matrix);
        if (r != 0) {
            return -1;
        }

        float min_max_diff = (max_matrix.buffer[0] - min_matrix.buffer[0]);
        /* Prevent divide by 0 by setting minimum value for divider */
        float row_scale = min_max_diff < 0.001 ? 1.0f : 1.0f / min_max_diff;

        r = subtract(&temp_matrix, min_matrix.buffer[0]);
        if (r != 0) {
            return -1;
        }

        r = scale(&temp_matrix, row_scale);
        if (r != 0) {
            return -1;
        }

        return 0;
    }

    /**
     * Calculate the standard deviation over a matrix on axis 0
     * @param input_matrix Input matrix (MxN)
     * @param output_matrix Output matrix (Nx1)
     * @returns 0 if OK
     */
    static int std_axis0(matrix_t *input_matrix, matrix_t *output_matrix) {
        if (input_matrix->cols != output_matrix->rows) {
            return -1;
        }

        if (output_matrix->cols != 1) {
            return -1;
        }

        for (size_t col = 0; col < input_matrix->cols; col++) {
            float sum = 0.0f;

            for (size_t row = 0; row < input_matrix->rows; row++) {
                sum += input_matrix->buffer[(row * input_matrix->cols) + col];
            }

            float mean = sum / input_matrix->rows;

            float std = 0.0f;
            float tmp;
            for (size_t row = 0; row < input_matrix->rows; row++) {
                tmp = input_matrix->buffer[(row * input_matrix->cols) + col] - mean;
                std += tmp * tmp;
            }

            output_matrix->buffer[col] = sqrt(std / input_matrix->rows);
        }

        return 0;
    }

    /**
     * Calculate the mean over a matrix on axis 0
     * @param input_matrix Input matrix (MxN)
     * @param output_matrix Output matrix (Nx1)
     * @returns 0 if OK
     */
    static int mean_axis0(matrix_t *input_matrix, matrix_t *output_matrix) {
        if (input_matrix->cols != output_matrix->rows) {
            return -1;
        }

        if (output_matrix->cols != 1) {
            return -1;
        }

        for (size_t col = 0; col < input_matrix->cols; col++) {
            // Note - not using CMSIS-DSP here
            // gathering up the current columnand moving it into sequential memory to use
            // SIMD to calculate the mean would take more time than the simple loop
            // so disable this case. The alternative is to use 2 transposes and on a "big" ARM
            // platform that will take more time

            float sum = 0.0f;

            for (size_t row = 0; row < input_matrix->rows; row++) {
                sum += input_matrix->buffer[( row * input_matrix->cols ) + col];
            }

            output_matrix->buffer[col] = sum / input_matrix->rows;
        }

        return 0;
    }

    /**
     * Pad an array.
     * Pads with the reflection of the vector mirrored along the edge of the array.
     * @param input Input matrix (MxN)
     * @param output Output matrix of size (M+pad_before+pad_after x N)
     * @param pad_before Number of items to pad before
     * @param pad_after Number of items to pad after
     * @returns 0 if OK
     */
    static int pad_1d_symmetric(matrix_t *input, matrix_t *output, uint16_t pad_before, uint16_t pad_after) {
        if (output->cols != input->cols) {
            return -1;
        }

        if (output->rows != input->rows + pad_before + pad_after) {
            return -1;
        }

        if (input->rows == 0) {
            return -1;
        }

        uint32_t pad_before_index = 0;
        bool pad_before_direction_up = true;

        for (int32_t ix = pad_before - 1; ix >= 0; ix--) {
            memcpy(output->buffer + (input->cols * ix),
                input->buffer + (pad_before_index * input->cols),
                input->cols * sizeof(float));

            if (pad_before_index == 0 && !pad_before_direction_up) {
                pad_before_direction_up = true;
            }
            else if (pad_before_index == input->rows - 1 && pad_before_direction_up) {
                pad_before_direction_up = false;
            }
            else if (pad_before_direction_up) {
                pad_before_index++;
            }
            else {
                pad_before_index--;
            }
        }

        memcpy(output->buffer + (input->cols * pad_before),
            input->buffer,
            input->rows * input->cols * sizeof(float));

        int32_t pad_after_index = input->rows - 1;
        bool pad_after_direction_up = false;

        for (int32_t ix = 0; ix < pad_after; ix++) {
            memcpy(output->buffer + (input->cols * (ix + pad_before + input->rows)),
                input->buffer + (pad_after_index * input->cols),
                input->cols * sizeof(float));

            if (pad_after_index == 0 && !pad_after_direction_up) {
                pad_after_direction_up = true;
            }
            else if (pad_after_index == static_cast<int32_t>(input->rows) - 1 && pad_after_direction_up) {
                pad_after_direction_up = false;
            }
            else if (pad_after_direction_up) {
                pad_after_index++;
            }
            else {
                pad_after_index--;
            }
        }

        return 0;
    }

    static int transform(float vector[], size_t len) {
    const size_t fft_data_out_size = (len / 2 + 1) * sizeof(fft_complex_t);
    const size_t fft_data_in_size = len * sizeof(float);

    // Allocate KissFFT input / output buffer
    fft_complex_t *fft_data_out =
        (fft_complex_t*)calloc(fft_data_out_size, 1);
    if (!fft_data_out) {
        return -1;
    }

    float *fft_data_in = (float*)calloc(fft_data_in_size, 1);
    if (!fft_data_in) {
        free(fft_data_out);
        return -1;
    }

    // Preprocess the input buffer with the data from the vector
    size_t halfLen = len / 2;
    for (size_t i = 0; i < halfLen; i++) {
        fft_data_in[i] = vector[i * 2];
        fft_data_in[len - 1 - i] = vector[i * 2 + 1];
    }
    if (len % 2 == 1) {
        fft_data_in[halfLen] = vector[len - 1];
    }

    int r = rfft(fft_data_in, len, fft_data_out, (len / 2 + 1), len);
    if (r != 0) {
        free(fft_data_in);
        free(fft_data_out);
        return r;
    }

    size_t i = 0;
    for (; i < len / 2 + 1; i++) {
        float temp = i * M_PI / (len * 2);
        vector[i] = fft_data_out[i].r * cos(temp) + fft_data_out[i].i * sin(temp);
    }
    //take advantage of hermetian symmetry to calculate remainder of signal
    for (; i < len; i++) {
        float temp = i * M_PI / (len * 2);
        int conj_idx = len-i;
        // second half bins not calculated would have just been the conjugate of the first half (note minus of imag)
        vector[i] = fft_data_out[conj_idx].r * cos(temp) - fft_data_out[conj_idx].i * sin(temp);
    }
    free(fft_data_in);
    free(fft_data_out);

    return 0;
}

    /**
     * Return the Discrete Cosine Transform of arbitrary type sequence 2.
     * @param input Input array (of size N)
     * @param N number of items in input and output array
     * @returns EIDSP_OK if OK
     */
    static int dct2(float *input, size_t N, DCT_NORMALIZATION_MODE normalization = DCT_NORMALIZATION_NONE) {
        if (N == 0) {
            return 0;
        }

        int ret = transform(input, N);
        if (ret != 0) {
            return -1;
        }

        // for some reason the output is 2x too low...
        for (size_t ix = 0; ix < N; ix++) {
            input[ix] *= 2;
        }

        if (normalization == DCT_NORMALIZATION_ORTHO) {
            input[0] = input[0] * sqrt(1.0f / static_cast<float>(4 * N));
            for (size_t ix = 1; ix < N; ix++) {
                input[ix] = input[ix] * sqrt(1.0f / static_cast<float>(2 * N));
            }
        }

        return 0;
    }

    /**
     * Discrete Cosine Transform of arbitrary type sequence 2 on a matrix.
     * @param matrix
     * @returns EIDSP_OK if OK
     */
    static int dct2(matrix_t *matrix, DCT_NORMALIZATION_MODE normalization = DCT_NORMALIZATION_NONE) {
        for (size_t row = 0; row < matrix->rows; row++) {
            int r = dct2(matrix->buffer + (row * matrix->cols), matrix->cols, normalization);
            if (r != 0) {
                return r;
            }
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
    static int rfft(const float *src, size_t src_size, fft_complex_t *output, size_t output_size, size_t n_fft) {
        size_t n_fft_out_features = (n_fft / 2) + 1;
        if (output_size != n_fft_out_features) {
            return -1;
        }

        // truncate if needed
        if (src_size > n_fft) {
            src_size = n_fft;
        }

        // declare input and output arrays
        float *fft_input_buffer = NULL;
        if (src_size == n_fft) {
            fft_input_buffer = (float*)src;
        }

        EI_DSP_MATRIX_B(fft_input, 1, n_fft, fft_input_buffer);
        if (!fft_input.buffer) {
            return -1;
        }

        if (!fft_input_buffer) {
            // copy from src to fft_input
            memcpy(fft_input.buffer, src, src_size * sizeof(float));
            // pad to the rigth with zeros
            memset(fft_input.buffer + src_size, 0, (n_fft - src_size) * sizeof(float));
        }

        if (n_fft != 32 && n_fft != 64 && n_fft != 128 && n_fft != 256 &&
            n_fft != 512 && n_fft != 1024 && n_fft != 2048 && n_fft != 4096) {
            int ret = software_rfft(fft_input.buffer, output, n_fft, n_fft_out_features);
            if (ret != 0) {
                return -1;
            }
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

            output[0].r = fft_output.buffer[0];
            output[0].i = 0.0f;
            output[n_fft_out_features - 1].r = fft_output.buffer[1];
            output[n_fft_out_features - 1].i = 0.0f;

            size_t fft_output_buffer_ix = 2;
            for (size_t ix = 1; ix < n_fft_out_features - 1; ix += 1) {
                output[ix].r = fft_output.buffer[fft_output_buffer_ix];
                output[ix].i = fft_output.buffer[fft_output_buffer_ix + 1];

                fft_output_buffer_ix += 2;
            }
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

    /**
     * Return evenly spaced numbers over a specified interval.
     * Returns num evenly spaced samples, calculated over the interval [start, stop].
     * The endpoint of the interval can optionally be excluded.
     *
     * Based on https://github.com/ntessore/algo/blob/master/linspace.c
     * Licensed in public domain (see LICENSE in repository above)
     *
     * @param start The starting value of the sequence.
     * @param stop The end value of the sequence.
     * @param number Number of samples to generate.
     * @param out Out array, with size `number`
     * @returns 0 if OK
     */
    static int linspace(float start, float stop, uint32_t number, float *out)
    {
        if (number < 1 || !out) {
            return -1;
        }

        if (number == 1) {
            out[0] = start;
            return 0;
        }

        // step size
        float step = (stop - start) / (number - 1);

        // do steps
        for (uint32_t ix = 0; ix < number - 1; ix++) {
            out[ix] = start + ix * step;
        }

        // last entry always stop
        out[number - 1] = stop;

        return 0;
    }

    private:
    static int software_rfft(float *fft_input, fft_complex_t *output, size_t n_fft, size_t n_fft_out_features)
    {
        // create fftr context
        size_t kiss_fftr_mem_length;

        kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, 0, NULL, NULL, &kiss_fftr_mem_length);
        if (!cfg) {
            return -1;
        }

        ei_dsp_register_alloc(kiss_fftr_mem_length, cfg);

        // execute the rfft operation
        kiss_fftr(cfg, fft_input, (kiss_fft_cpx*)output);

        free(cfg);

        return 0;
    }

    static int software_rfft(float *fft_input, float *output, size_t n_fft, size_t n_fft_out_features) {
        kiss_fft_cpx *fft_output = (kiss_fft_cpx*)malloc(n_fft_out_features * sizeof(kiss_fft_cpx));
        if (!fft_output) {
            return -1;
        }

        size_t kiss_fftr_mem_length;

        // create fftr context
        kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, 0, NULL, NULL, &kiss_fftr_mem_length);
        if (!cfg) {
            free(fft_output);
            return -1;
        }

        ei_dsp_register_alloc(kiss_fftr_mem_length, cfg);

        // execute the rfft operation
        kiss_fftr(cfg, fft_input, fft_output);

        // and write back to the output
        for (size_t ix = 0; ix < n_fft_out_features; ix++) {
            output[ix] = sqrt(pow(fft_output[ix].r, 2) + pow(fft_output[ix].i, 2));
        }

        free(cfg);
        free(fft_output);

        return 0;
    }

    // more numpy code here
};

// !!---------------------------------------------- NUMPY CLASS END ----------------------------------------------!!
// !!---------------------------------------------- NUMPY CLASS END ----------------------------------------------!!


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

class functions {
    public:

    /**
     * This function handle the issue with zero values if the are exposed
     * to become an argument for any log function.
     * @param input Array
     * @param input_size Size of array
     * @returns void
     */
    static void zero_handling(float *input, size_t input_size) {
        for (size_t ix = 0; ix < input_size; ix++) {
            if (input[ix] == 0) {
                input[ix] = 1e-10;
            }
        }
    }

    /**
     * This function handle the issue with zero values if the are exposed
     * to become an argument for any log function.
     * @param input Matrix
     * @returns void
     */
    static void zero_handling(matrix_t *input) {
        zero_handling(input->buffer, input->rows * input->cols);
    }

        /**
     * Converting from frequency to Mel scale
     *
     * @param f The frequency values(or a single frequency) in Hz.
     * @returns The mel scale values(or a single mel).
     */
    static float frequency_to_mel(float f) {
        return 1127.0 * numpy::log(1 + f / 700.0f);
    }

    /**
     * Converting from Mel scale to frequency.
     *
     * @param mel The mel scale values(or a single mel).
     * @returns The frequency values(or a single frequency) in Hz.
     */
    static float mel_to_frequency(float mel) {
        return 700.0f * (exp(mel / 1127.0f) - 1.0f);
    }

    /**
     * Triangle, I'm not really sure what this does
     * @param x Linspace output, will be overwritten!
     * @param x_size Size of the linspace output
     * @param left
     * @param middle
     * @param right
     */
    static int triangle(float *x, size_t x_size, int left, int middle, int right) {
        EI_DSP_MATRIX(out, 1, x_size);

        for (size_t ix = 0; ix < x_size; ix++) {
            if (x[ix] > left && x[ix] <= middle) {
                out.buffer[ix] = (x[ix] - left) / (middle - left);
            }

            if (x[ix] < right && middle <= x[ix]) {
                out.buffer[ix] = (right - x[ix]) / (right - middle);
            }
        }

        memcpy(x, out.buffer, x_size * sizeof(float));

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
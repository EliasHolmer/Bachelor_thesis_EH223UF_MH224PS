#ifndef _INFERENCE_H
#define _INFERENCE_H

// Undefine min/max macros as these conflict with C++ std min/max functions
// these are often included by Arduino cores
#include <Arduino.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "edge-impulse-sdk/tensorflow/lite/kernels/internal/types.h"
#include "edge-impulse-sdk/dsp/spectral/spectral.hpp"
#include "edge-impulse-sdk/dsp/speechpy/speechpy.hpp"

#ifdef min
#undef min
#endif // min
#ifdef max
#undef max
#endif // max
#ifdef round
#undef round
#endif // round
// Similar the ESP32 seems to define this, which is also used as an enum value in TFLite
#ifdef DEFAULT
#undef DEFAULT
#endif // DEFAULT
// Infineon core defines this, conflicts with CMSIS/DSP/Include/dsp/controller_functions.h
#ifdef A0
#undef A0
#endif // A0
#ifdef A1
#undef A1
#endif // A1
#ifdef A2
#undef A2
#endif // A2

extern void ei_printf(const char *format, ...);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of Model_metadata.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_MODEL_METADATA_H_
#define _EI_CLASSIFIER_MODEL_METADATA_H_

#define EI_CLASSIFIER_NONE 255
#define EI_CLASSIFIER_UTENSOR 1
#define EI_CLASSIFIER_TFLITE 2
#define EI_CLASSIFIER_CUBEAI 3
#define EI_CLASSIFIER_TFLITE_FULL 4
#define EI_CLASSIFIER_TENSAIFLOW 5
#define EI_CLASSIFIER_TENSORRT 6

#define EI_CLASSIFIER_SENSOR_UNKNOWN -1
#define EI_CLASSIFIER_SENSOR_MICROPHONE 1
#define EI_CLASSIFIER_SENSOR_ACCELEROMETER 2
#define EI_CLASSIFIER_SENSOR_CAMERA 3
#define EI_CLASSIFIER_SENSOR_9DOF 4
#define EI_CLASSIFIER_SENSOR_ENVIRONMENTAL 5
#define EI_CLASSIFIER_SENSOR_FUSION 6

// These must match the enum values in TensorFlow Lite's "TfLiteType"
#define EI_CLASSIFIER_DATATYPE_FLOAT32 1
#define EI_CLASSIFIER_DATATYPE_INT8 9

#define EI_CLASSIFIER_PROJECT_ID 94516
#define EI_CLASSIFIER_PROJECT_OWNER "Mattias"
#define EI_CLASSIFIER_PROJECT_NAME "bees2.0"
#define EI_CLASSIFIER_PROJECT_DEPLOY_VERSION 17
#define EI_CLASSIFIER_NN_INPUT_FRAME_SIZE 2480
#define EI_CLASSIFIER_RAW_SAMPLE_COUNT 32000
#define EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME 1
#define EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE (EI_CLASSIFIER_RAW_SAMPLE_COUNT * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME)
#define EI_CLASSIFIER_INPUT_WIDTH 0
#define EI_CLASSIFIER_INPUT_HEIGHT 0
#define EI_CLASSIFIER_INPUT_FRAMES 0
#define EI_CLASSIFIER_INTERVAL_MS 0.0625
#define EI_CLASSIFIER_LABEL_COUNT 2
#define EI_CLASSIFIER_HAS_ANOMALY 0
#define EI_CLASSIFIER_FREQUENCY 16000
#define EI_CLASSIFIER_USE_QUANTIZED_DSP_BLOCK 0
#define EI_CLASSIFIER_HAS_MODEL_VARIABLES 1

#define EI_CLASSIFIER_OBJECT_DETECTION 0
#define EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR 0

#define EI_CLASSIFIER_TFLITE_INPUT_DATATYPE EI_CLASSIFIER_DATATYPE_INT8
#define EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED 1
#define EI_CLASSIFIER_TFLITE_INPUT_SCALE 0.0030484069138765335
#define EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT -128
#define EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE EI_CLASSIFIER_DATATYPE_INT8
#define EI_CLASSIFIER_TFLITE_OUTPUT_QUANTIZED 1
#define EI_CLASSIFIER_TFLITE_OUTPUT_SCALE 0.00390625
#define EI_CLASSIFIER_TFLITE_OUTPUT_ZEROPOINT -128

#define EI_CLASSIFIER_INFERENCING_ENGINE EI_CLASSIFIER_TFLITE

#define EI_CLASSIFIER_COMPILED 1
#define EI_CLASSIFIER_HAS_TFLITE_OPS_RESOLVER 1

#define EI_CLASSIFIER_HAS_FFT_INFO 1
#define EI_CLASSIFIER_LOAD_FFT_32 0
#define EI_CLASSIFIER_LOAD_FFT_64 0
#define EI_CLASSIFIER_LOAD_FFT_128 0
#define EI_CLASSIFIER_LOAD_FFT_256 1
#define EI_CLASSIFIER_LOAD_FFT_512 0
#define EI_CLASSIFIER_LOAD_FFT_1024 0
#define EI_CLASSIFIER_LOAD_FFT_2048 0
#define EI_CLASSIFIER_LOAD_FFT_4096 0

#define EI_CLASSIFIER_SENSOR EI_CLASSIFIER_SENSOR_MICROPHONE
#define EI_CLASSIFIER_FUSION_AXES_STRING "audio"

#ifndef EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4
#endif // EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW
#define EI_CLASSIFIER_SLICE_SIZE (EI_CLASSIFIER_RAW_SAMPLE_COUNT / EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)

typedef struct
{
    uint16_t implementation_version;
    int axes;
    float scale_axes;
    bool average;
    bool minimum;
    bool maximum;
    bool rms;
    bool stdev;
    bool skewness;
    bool kurtosis;
} ei_dsp_config_flatten_t;

typedef struct
{
    uint16_t implementation_version;
    int axes;
    const char *channels;
} ei_dsp_config_image_t;

typedef struct
{
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
} ei_dsp_config_mfcc_t;

typedef struct
{
    uint16_t implementation_version;
    int axes;
    float frame_length;
    float frame_stride;
    int num_filters;
    int fft_length;
    int low_frequency;
    int high_frequency;
    int win_size;
    int noise_floor_db;
} ei_dsp_config_mfe_t;

typedef struct
{
    uint16_t implementation_version;
    int axes;
    float scale_axes;
} ei_dsp_config_raw_t;

typedef struct
{
    uint16_t implementation_version;
    int axes;
    float scale_axes;
    const char *filter_type;
    float filter_cutoff;
    int filter_order;
    int fft_length;
    int spectral_peaks_count;
    float spectral_peaks_threshold;
    const char *spectral_power_edges;
} ei_dsp_config_spectral_analysis_t;

typedef struct
{
    uint16_t implementation_version;
    int axes;
    float frame_length;
    float frame_stride;
    int fft_length;
    int noise_floor_db;
    bool show_axes;
} ei_dsp_config_spectrogram_t;

typedef struct
{
    uint16_t implementation_version;
    int axes;
    float frame_length;
    float frame_stride;
    int num_filters;
    int fft_length;
    int low_frequency;
    int high_frequency;
    float pre_cof;
} ei_dsp_config_audio_syntiant_t;

#endif // _EI_CLASSIFIER_MODEL_METADATA_H_

// End of  Model_metadata.h ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Start of ei_model_types.h  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EDGE_IMPULSE_MODEL_TYPES_H_
#define _EDGE_IMPULSE_MODEL_TYPES_H_

typedef struct
{
    size_t n_output_features;
    int (*extract_fn)(ei::signal_t *signal, ei::matrix_t *output_matrix, void *config, const float frequency);
    void *config;
    uint8_t *axes;
    size_t axes_size;
} ei_model_dsp_t;

typedef struct
{
    uint16_t implementation_version;
    uint32_t average_window_duration_ms;
    float detection_threshold;
    uint32_t suppression_ms;
    uint32_t suppression_flags;
} ei_model_performance_calibration_t;

#endif // _EDGE_IMPULSE_MODEL_TYPES_H_

// End of ei_model_types.h  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of model-parameters/model_variables.h -------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_MODEL_VARIABLES_H_
#define _EI_CLASSIFIER_MODEL_VARIABLES_H_

const char *ei_classifier_inferencing_categories[] = {"Bee", "notBee"};

uint8_t ei_dsp_config_52_axes[] = {0};
const uint32_t ei_dsp_config_52_axes_size = 1;
ei_dsp_config_mfe_t ei_dsp_config_52 = {
    3,
    1,
    0.032f,
    0.032f,
    40,
    256,
    300,
    0,
    101,
    -72};

#endif // _EI_CLASSIFIER_MODEL_METADATA_H_

// End of model-parameters/model_variables.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_signal_with_range.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_SIGNAL_WITH_RANGE_H_
#define _EI_CLASSIFIER_SIGNAL_WITH_RANGE_H_

#if !EIDSP_SIGNAL_C_FN_POINTER

using namespace ei;

class SignalWithRange
{
public:
    SignalWithRange(signal_t *original_signal, uint32_t range_start, uint32_t range_end) : _original_signal(original_signal), _range_start(range_start), _range_end(range_end)
    {
    }

    signal_t *get_signal()
    {
        if (this->_range_start == 0 && this->_range_end == this->_original_signal->total_length)
        {
            return this->_original_signal;
        }
        wrapped_signal.total_length = _range_end - _range_start;
        wrapped_signal.get_data = mbed::callback(this, &SignalWithRange::get_data);
        return &wrapped_signal;
    }

    int get_data(size_t offset, size_t length, float *out_ptr)
    {
        return _original_signal->get_data(offset + _range_start, length, out_ptr);
    }

private:
    signal_t *_original_signal;
    uint32_t _range_start;
    uint32_t _range_end;
    signal_t wrapped_signal;
};

#endif // #if !EIDSP_SIGNAL_C_FN_POINTER

#endif // _EI_CLASSIFIER_SIGNAL_WITH_RANGE_H_

// End of ei_signal_with_range.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TENSORFLOW_LITE_C_COMMON_H_
#define TENSORFLOW_LITE_C_COMMON_H_


#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    // The list of external context types known to TF Lite. This list exists solely
    // to avoid conflicts and to ensure ops can share the external contexts they
    // need. Access to the external contexts is controlled by one of the
    // corresponding support files.
    typedef enum TfLiteExternalContextType
    {
        kTfLiteEigenContext = 0,      // include eigen_support.h to use.
        kTfLiteGemmLowpContext = 1,   // include gemm_support.h to use.
        kTfLiteEdgeTpuContext = 2,    // Placeholder for Edge TPU support.
        kTfLiteCpuBackendContext = 3, // include cpu_backend_context.h to use.
        kTfLiteMaxExternalContexts = 4
    } TfLiteExternalContextType;

    // Forward declare so dependent structs and methods can reference these types
    // prior to the struct definitions.
    struct TfLiteContext;
    struct TfLiteDelegate;
    struct TfLiteRegistration;

    // An external context is a collection of information unrelated to the TF Lite
    // framework, but useful to a subset of the ops. TF Lite knows very little
    // about the actual contexts, but it keeps a list of them, and is able to
    // refresh them if configurations like the number of recommended threads
    // change.
    typedef struct TfLiteExternalContext
    {
        TfLiteExternalContextType type;
        TfLiteStatus (*Refresh)(struct TfLiteContext *context);
    } TfLiteExternalContext;

#define kTfLiteOptionalTensor (-1)

    // Fixed size list of integers. Used for dimensions and inputs/outputs tensor
    // indices
    typedef struct TfLiteIntArray
    {
        int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
#if (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
     __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                           \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
        int data[0];
#else
    int data[];
#endif
    } TfLiteIntArray;

    // Given the size (number of elements) in a TfLiteIntArray, calculate its size
    // in bytes.
    int TfLiteIntArrayGetSizeInBytes(int size);

#ifndef TF_LITE_STATIC_MEMORY
    // Create a array of a given `size` (uninitialized entries).
    // This returns a pointer, that you must free using TfLiteIntArrayFree().
    TfLiteIntArray *TfLiteIntArrayCreate(int size);
#endif

    // Check if two intarrays are equal. Returns 1 if they are equal, 0 otherwise.
    int TfLiteIntArrayEqual(const TfLiteIntArray *a, const TfLiteIntArray *b);

    // Check if an intarray equals an array. Returns 1 if equals, 0 otherwise.
    int TfLiteIntArrayEqualsArray(const TfLiteIntArray *a, int b_size,
                                  const int b_data[]);

#ifndef TF_LITE_STATIC_MEMORY
    // Create a copy of an array passed as `src`.
    // You are expected to free memory with TfLiteIntArrayFree
    TfLiteIntArray *TfLiteIntArrayCopy(const TfLiteIntArray *src);

    // Free memory of array `a`.
    void TfLiteIntArrayFree(TfLiteIntArray *a);
#endif // TF_LITE_STATIC_MEMORY

    // Fixed size list of floats. Used for per-channel quantization.
    typedef struct TfLiteFloatArray
    {
        int size;
        // gcc 6.1+ have a bug where flexible members aren't properly handled
        // https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
        // This also applies to the toolchain used for Qualcomm Hexagon DSPs.
        float data[];
    } TfLiteFloatArray;

    // Given the size (number of elements) in a TfLiteFloatArray, calculate its size
    // in bytes.
    int TfLiteFloatArrayGetSizeInBytes(int size);

#ifndef TF_LITE_STATIC_MEMORY
    // Create a array of a given `size` (uninitialized entries).
    // This returns a pointer, that you must free using TfLiteFloatArrayFree().
    TfLiteFloatArray *TfLiteFloatArrayCreate(int size);

    // Free memory of array `a`.
    void TfLiteFloatArrayFree(TfLiteFloatArray *a);
#endif // TF_LITE_STATIC_MEMORY

// Since we must not depend on any libraries, define a minimal subset of
// error macros while avoiding names that have pre-conceived meanings like
// assert and check.

// Try to make all reporting calls through TF_LITE_KERNEL_LOG rather than
// calling the context->ReportError function directly, so that message strings
// can be stripped out if the binary size needs to be severely optimized.
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)                \
    do                                                  \
    {                                                   \
        (context)->ReportError((context), __VA_ARGS__); \
    } while (false)

#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)              \
    do                                                      \
    {                                                       \
        if ((context) != nullptr)                           \
        {                                                   \
            (context)->ReportError((context), __VA_ARGS__); \
        }                                                   \
    } while (false)
#else // TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)
#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)
#endif // TF_LITE_STRIP_ERROR_STRINGS

// Check whether value is true, and if not return kTfLiteError from
// the current function (and report the error string msg).
#define TF_LITE_ENSURE_MSG(context, value, msg)              \
    do                                                       \
    {                                                        \
        if (!(value))                                        \
        {                                                    \
            TF_LITE_KERNEL_LOG((context), __FILE__ " " msg); \
            return kTfLiteError;                             \
        }                                                    \
    } while (0)

// Check whether the value `a` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
#define TF_LITE_ENSURE(context, a)                                            \
    do                                                                        \
    {                                                                         \
        if (!(a))                                                             \
        {                                                                     \
            TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                               __LINE__, #a);                                 \
            return kTfLiteError;                                              \
        }                                                                     \
    } while (0)

#define TF_LITE_ENSURE_STATUS(a)    \
    do                              \
    {                               \
        const TfLiteStatus s = (a); \
        if (s != kTfLiteOk)         \
        {                           \
            return s;               \
        }                           \
    } while (0)

// Check whether the value `a == b` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
// `a` and `b` may be evaluated more than once, so no side effects or
// extremely expensive computations should be done.
// NOTE: Use TF_LITE_ENSURE_TYPES_EQ if comparing TfLiteTypes.
#define TF_LITE_ENSURE_EQ(context, a, b)                                         \
    do                                                                           \
    {                                                                            \
        if ((a) != (b))                                                          \
        {                                                                        \
            TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                               __LINE__, #a, #b, (a), (b));                      \
            return kTfLiteError;                                                 \
        }                                                                        \
    } while (0)

#define TF_LITE_ENSURE_TYPES_EQ(context, a, b)                                   \
    do                                                                           \
    {                                                                            \
        if ((a) != (b))                                                          \
        {                                                                        \
            TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%s != %s)", __FILE__, \
                               __LINE__, #a, #b, TfLiteTypeGetName(a),           \
                               TfLiteTypeGetName(b));                            \
            return kTfLiteError;                                                 \
        }                                                                        \
    } while (0)

#define TF_LITE_ENSURE_NEAR(context, a, b, epsilon)                                \
    do                                                                             \
    {                                                                              \
        auto delta = ((a) > (b)) ? ((a) - (b)) : ((b) - (a));                      \
        if (delta > epsilon)                                                       \
        {                                                                          \
            TF_LITE_KERNEL_LOG((context), "%s:%d %s not near %s (%f != %f)",       \
                               __FILE__, __LINE__, #a, #b, static_cast<double>(a), \
                               static_cast<double>(b));                            \
            return kTfLiteError;                                                   \
        }                                                                          \
    } while (0)

#define TF_LITE_ENSURE_OK(context, status) \
    do                                     \
    {                                      \
        const TfLiteStatus s = (status);   \
        if ((s) != kTfLiteOk)              \
        {                                  \
            return s;                      \
        }                                  \
    } while (0)

    // Single-precision complex data type compatible with the C99 definition.
    typedef struct TfLiteComplex64
    {
        float re, im; // real and imaginary parts, respectively.
    } TfLiteComplex64;

    // Double-precision complex data type compatible with the C99 definition.
    typedef struct TfLiteComplex128
    {
        double re, im; // real and imaginary parts, respectively.
    } TfLiteComplex128;

    // Half precision data type compatible with the C99 definition.
    typedef struct TfLiteFloat16
    {
        uint16_t data;
    } TfLiteFloat16;

    // Return the name of a given type, for error reporting purposes.
    const char *TfLiteTypeGetName(TfLiteType type);

    // SupportedQuantizationTypes.
    typedef enum TfLiteQuantizationType
    {
        // No quantization.
        kTfLiteNoQuantization = 0,
        // Affine quantization (with support for per-channel quantization).
        // Corresponds to TfLiteAffineQuantization.
        kTfLiteAffineQuantization = 1,
    } TfLiteQuantizationType;

    // Structure specifying the quantization used by the tensor, if-any.
    typedef struct TfLiteQuantization
    {
        // The type of quantization held by params.
        TfLiteQuantizationType type;
        // Holds an optional reference to a quantization param structure. The actual
        // type depends on the value of the `type` field (see the comment there for
        // the values and corresponding types).
        void *params;
    } TfLiteQuantization;

    // Parameters for asymmetric quantization across a dimension (i.e per output
    // channel quantization).
    // quantized_dimension specifies which dimension the scales and zero_points
    // correspond to.
    // For a particular value in quantized_dimension, quantized values can be
    // converted back to float using:
    //     real_value = scale * (quantized_value - zero_point)
    typedef struct TfLiteAffineQuantization
    {
        TfLiteFloatArray *scale;
        TfLiteIntArray *zero_point;
        int32_t quantized_dimension;
    } TfLiteAffineQuantization;

    /* A union of pointers that points to memory for a given tensor. */
    typedef union TfLitePtrUnion
    {
        /* Do not access these members directly, if possible, use
         * GetTensorData<TYPE>(tensor) instead, otherwise only access .data, as other
         * members are deprecated. */
        int32_t *i32;
        uint32_t *u32;
        int64_t *i64;
        uint64_t *u64;
        float *f;
        TfLiteFloat16 *f16;
        double *f64;
        char *raw;
        const char *raw_const;
        uint8_t *uint8;
        bool *b;
        int16_t *i16;
        TfLiteComplex64 *c64;
        TfLiteComplex128 *c128;
        int8_t *int8;
        /* Only use this member. */
        void *data;
    } TfLitePtrUnion;

    // Memory allocation strategies.
    //  * kTfLiteMmapRo: Read-only memory-mapped data, or data externally allocated.
    //  * kTfLiteArenaRw: Arena allocated with no guarantees about persistence,
    //        and available during eval.
    //  * kTfLiteArenaRwPersistent: Arena allocated but persistent across eval, and
    //        only available during eval.
    //  * kTfLiteDynamic: Allocated during eval, or for string tensors.
    //  * kTfLitePersistentRo: Allocated and populated during prepare. This is
    //        useful for tensors that can be computed during prepare and treated
    //        as constant inputs for downstream ops (also in prepare).
    //  * kTfLiteCustom: Custom memory allocation provided by the user. See
    //        TfLiteCustomAllocation below.
    typedef enum TfLiteAllocationType
    {
        kTfLiteMemNone = 0,
        kTfLiteMmapRo,
        kTfLiteArenaRw,
        kTfLiteArenaRwPersistent,
        kTfLiteDynamic,
        kTfLitePersistentRo,
        kTfLiteCustom,
    } TfLiteAllocationType;

    // The delegates should use zero or positive integers to represent handles.
    // -1 is reserved from unallocated status.
    typedef int TfLiteBufferHandle;
    enum
    {
        kTfLiteNullBufferHandle = -1,
    };

    // Storage format of each dimension in a sparse tensor.
    typedef enum TfLiteDimensionType
    {
        kTfLiteDimDense = 0,
        kTfLiteDimSparseCSR,
    } TfLiteDimensionType;

    // Metadata to encode each dimension in a sparse tensor.
    typedef struct TfLiteDimensionMetadata
    {
        TfLiteDimensionType format;
        int dense_size;
        TfLiteIntArray *array_segments;
        TfLiteIntArray *array_indices;
    } TfLiteDimensionMetadata;

    // Parameters used to encode a sparse tensor. For detailed explanation of each
    // field please refer to lite/schema/schema.fbs.
    typedef struct TfLiteSparsity
    {
        TfLiteIntArray *traversal_order;
        TfLiteIntArray *block_map;
        TfLiteDimensionMetadata *dim_metadata;
        int dim_metadata_size;
    } TfLiteSparsity;

    // Defines a custom memory allocation not owned by the runtime.
    // `data` should be aligned to kDefaultTensorAlignment defined in
    // lite/util.h. (Currently 64 bytes)
    // NOTE: See Interpreter.SetCustomAllocationForTensor for details on usage.
    typedef struct TfLiteCustomAllocation
    {
        void *data;
        size_t bytes;
    } TfLiteCustomAllocation;

    // The flags used in `Interpreter::SetCustomAllocationForTensor`.
    // Note that this is a bitmask, so the values should be 1, 2, 4, 8, ...etc.
    typedef enum TfLiteCustomAllocationFlags
    {
        kTfLiteCustomAllocationFlagsNone = 0,
        // Skips checking whether allocation.data points to an aligned buffer as
        // expected by the TFLite runtime.
        // NOTE: Setting this flag can cause crashes when calling Invoke().
        // Use with caution.
        kTfLiteCustomAllocationFlagsSkipAlignCheck = 1,
    } TfLiteCustomAllocationFlags;

    // A tensor in the interpreter system which is a wrapper around a buffer of
    // data including a dimensionality (or NULL if not currently defined).
    typedef struct TfLiteTensor
    {
        // The data type specification for data stored in `data`. This affects
        // what member of `data` union should be used.
        TfLiteType type;
        // A union of data pointers. The appropriate type should be used for a typed
        // tensor based on `type`.
        TfLitePtrUnion data;
        // A pointer to a structure representing the dimensionality interpretation
        // that the buffer should have. NOTE: the product of elements of `dims`
        // and the element datatype size should be equal to `bytes` below.
        TfLiteIntArray *dims;
        // Quantization information.
        TfLiteQuantizationParams params;
        // How memory is mapped
        //  kTfLiteMmapRo: Memory mapped read only.
        //  i.e. weights
        //  kTfLiteArenaRw: Arena allocated read write memory
        //  (i.e. temporaries, outputs).
        TfLiteAllocationType allocation_type;
        // The number of bytes required to store the data of this Tensor. I.e.
        // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
        // type is kTfLiteFloat32 and dims = {3, 2} then
        // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
        size_t bytes;

        // An opaque pointer to a tflite::MMapAllocation
        const void *allocation;

        // Null-terminated name of this tensor.
        const char *name;

        // The delegate which knows how to handle `buffer_handle`.
        // WARNING: This is an experimental interface that is subject to change.
        struct TfLiteDelegate *delegate;

        // An integer buffer handle that can be handled by `delegate`.
        // The value is valid only when delegate is not null.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteBufferHandle buffer_handle;

        // If the delegate uses its own buffer (e.g. GPU memory), the delegate is
        // responsible to set data_is_stale to true.
        // `delegate->CopyFromBufferHandle` can be called to copy the data from
        // delegate buffer.
        // WARNING: This is an // experimental interface that is subject to change.
        bool data_is_stale;

        // True if the tensor is a variable.
        bool is_variable;

        // Quantization information. Replaces params field above.
        TfLiteQuantization quantization;

        // Parameters used to encode a sparse tensor.
        // This is optional. The field is NULL if a tensor is dense.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteSparsity *sparsity;

        // Optional. Encodes shapes with unknown dimensions with -1. This field is
        // only populated when unknown dimensions exist in a read-write tensor (i.e.
        // an input or output tensor). (e.g.  `dims` contains [1, 1, 1, 3] and
        // `dims_signature` contains [1, -1, -1, 3]).
        const TfLiteIntArray *dims_signature;
    } TfLiteTensor;

    // A structure representing an instance of a node.
    // This structure only exhibits the inputs, outputs and user defined data, not
    // other features like the type.
    typedef struct TfLiteNode
    {
        // Inputs to this node expressed as indices into the simulator's tensors.
        TfLiteIntArray *inputs;

        // Outputs to this node expressed as indices into the simulator's tensors.
        TfLiteIntArray *outputs;

        // intermediate tensors to this node expressed as indices into the simulator's
        // tensors.
        TfLiteIntArray *intermediates;

        // Temporary tensors uses during the computations. This usually contains no
        // tensors, but ops are allowed to change that if they need scratch space of
        // any sort.
        TfLiteIntArray *temporaries;

        // Opaque data provided by the node implementer through `Registration.init`.
        void *user_data;

        // Opaque data provided to the node if the node is a builtin. This is usually
        // a structure defined in builtin_op_data.h
        void *builtin_data;

        // Custom initial data. This is the opaque data provided in the flatbuffer.
        // WARNING: This is an experimental interface that is subject to change.
        const void *custom_initial_data;
        int custom_initial_data_size;

        // The pointer to the delegate. This is non-null only when the node is
        // created by calling `interpreter.ModifyGraphWithDelegate`.
        // WARNING: This is an experimental interface that is subject to change.
        struct TfLiteDelegate *delegate;
    } TfLiteNode;

    // Light-weight tensor struct for TF Micro runtime. Provides the minimal amount
    // of information required for a kernel to run during TfLiteRegistration::Eval.
    // TODO(b/160955687): Move this field into TF_LITE_STATIC_MEMORY when TFLM
    // builds with this flag by default internally.
    typedef struct TfLiteEvalTensor
    {
        // A union of data pointers. The appropriate type should be used for a typed
        // tensor based on `type`.
        TfLitePtrUnion data;

        // A pointer to a structure representing the dimensionality interpretation
        // that the buffer should have.
        TfLiteIntArray *dims;

        // The data type specification for data stored in `data`. This affects
        // what member of `data` union should be used.
        TfLiteType type;
    } TfLiteEvalTensor;

#ifndef TF_LITE_STATIC_MEMORY
    // Free data memory of tensor `t`.
    void TfLiteTensorDataFree(TfLiteTensor *t);

    // Free quantization data.
    void TfLiteQuantizationFree(TfLiteQuantization *quantization);

    // Free sparsity parameters.
    void TfLiteSparsityFree(TfLiteSparsity *sparsity);

    // Free memory of tensor `t`.
    void TfLiteTensorFree(TfLiteTensor *t);

    // Set all of a tensor's fields (and free any previously allocated data).
    void TfLiteTensorReset(TfLiteType type, const char *name, TfLiteIntArray *dims,
                           TfLiteQuantizationParams quantization, char *buffer,
                           size_t size, TfLiteAllocationType allocation_type,
                           const void *allocation, bool is_variable,
                           TfLiteTensor *tensor);

    // Resize the allocated data of a (dynamic) tensor. Tensors with allocation
    // types other than kTfLiteDynamic will be ignored.
    void TfLiteTensorRealloc(size_t num_bytes, TfLiteTensor *tensor);
#endif // TF_LITE_STATIC_MEMORY

    // WARNING: This is an experimental interface that is subject to change.
    //
    // Currently, TfLiteDelegateParams has to be allocated in a way that it's
    // trivially destructable. It will be stored as `builtin_data` field in
    // `TfLiteNode` of the delegate node.
    //
    // See also the `CreateDelegateParams` function in `interpreter.cc` details.
    typedef struct TfLiteDelegateParams
    {
        struct TfLiteDelegate *delegate;
        TfLiteIntArray *nodes_to_replace;
        TfLiteIntArray *input_tensors;
        TfLiteIntArray *output_tensors;
    } TfLiteDelegateParams;

    typedef struct TfLiteContext
    {
        // Number of tensors in the context.
        size_t tensors_size;

        // The execution plan contains a list of the node indices in execution
        // order. execution_plan->size is the current number of nodes. And,
        // execution_plan->data[0] is the first node that needs to be run.
        // TfLiteDelegates can traverse the current execution plan by iterating
        // through each member of this array and using GetNodeAndRegistration() to
        // access details about a node. i.e.
        // TfLiteIntArray* execution_plan;
        // TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
        // for (int exec_index = 0; exec_index < execution_plan->size; exec_index++) {
        //    int node_index = execution_plan->data[exec_index];
        //    TfLiteNode* node;
        //    TfLiteRegistration* reg;
        //    context->GetNodeAndRegistration(context, node_index, &node, &reg);
        // }
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteStatus (*GetExecutionPlan)(struct TfLiteContext *context,
                                         TfLiteIntArray **execution_plan);

        // An array of tensors in the interpreter context (of length `tensors_size`)
        TfLiteTensor *tensors;

        // opaque full context ptr (an opaque c++ data structure)
        void *impl_;

        // Request memory pointer be resized. Updates dimensions on the tensor.
        // NOTE: ResizeTensor takes ownership of newSize.
        TfLiteStatus (*ResizeTensor)(struct TfLiteContext *, TfLiteTensor *tensor,
                                     TfLiteIntArray *new_size);
        // Request that an error be reported with format string msg.
        void (*ReportError)(struct TfLiteContext *, const char *msg, ...);

        // Add `tensors_to_add` tensors, preserving pre-existing Tensor entries.  If
        // non-null, the value pointed to by `first_new_tensor_index` will be set to
        // the index of the first new tensor.
        TfLiteStatus (*AddTensors)(struct TfLiteContext *, int tensors_to_add,
                                   int *first_new_tensor_index);

        // Get a Tensor node by node_index.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteStatus (*GetNodeAndRegistration)(
            struct TfLiteContext *, int node_index, TfLiteNode **node,
            struct TfLiteRegistration **registration);

        // Replace ops with one or more stub delegate operations. This function
        // does not take ownership of `nodes_to_replace`.
        TfLiteStatus (*ReplaceNodeSubsetsWithDelegateKernels)(
            struct TfLiteContext *, struct TfLiteRegistration registration,
            const TfLiteIntArray *nodes_to_replace, struct TfLiteDelegate *delegate);

        // Number of threads that are recommended to subsystems like gemmlowp and
        // eigen.
        int recommended_num_threads;

        // Access external contexts by type.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteExternalContext *(*GetExternalContext)(struct TfLiteContext *,
                                                     TfLiteExternalContextType);
        // Set the value of a external context. Does not take ownership of the
        // pointer.
        // WARNING: This is an experimental interface that is subject to change.
        void (*SetExternalContext)(struct TfLiteContext *, TfLiteExternalContextType,
                                   TfLiteExternalContext *);

        // Flag for allowing float16 precision for FP32 calculation.
        // default: false.
        // WARNING: This is an experimental API and subject to change.
        bool allow_fp32_relax_to_fp16;

        // Pointer to the op-level profiler, if set; nullptr otherwise.
        void *profiler;

        // Allocate persistent buffer which has the same life time as the interpreter.
        // Returns nullptr on failure.
        // The memory is allocated from heap for TFL, and from tail in TFLM.
        // This method is only available in Init or Prepare stage.
        // WARNING: This is an experimental interface that is subject to change.
        void *(*AllocatePersistentBuffer)(struct TfLiteContext *ctx, size_t bytes);

        // Allocate a buffer which will be deallocated right after invoke phase.
        // The memory is allocated from heap in TFL, and from volatile arena in TFLM.
        // This method is only available in invoke stage.
        // NOTE: If possible use RequestScratchBufferInArena method to avoid memory
        // allocation during inference time.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteStatus (*AllocateBufferForEval)(struct TfLiteContext *ctx, size_t bytes,
                                              void **ptr);

        // Request a scratch buffer in the arena through static memory planning.
        // This method is only available in Prepare stage and the buffer is allocated
        // by the interpreter between Prepare and Eval stage. In Eval stage,
        // GetScratchBuffer API can be used to fetch the address.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteStatus (*RequestScratchBufferInArena)(struct TfLiteContext *ctx,
                                                    size_t bytes, int *buffer_idx);

        // Get the scratch buffer pointer.
        // This method is only available in Eval stage.
        // WARNING: This is an experimental interface that is subject to change.
        void *(*GetScratchBuffer)(struct TfLiteContext *ctx, int buffer_idx);

        // Resize the memory pointer of the `tensor`. This method behaves the same as
        // `ResizeTensor`, except that it makes a copy of the shape array internally
        // so the shape array could be deallocated right afterwards.
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteStatus (*ResizeTensorExplicit)(struct TfLiteContext *ctx,
                                             TfLiteTensor *tensor, int dims,
                                             const int *shape);

        // This method provides a preview of post-delegation partitioning. Each
        // TfLiteDelegateParams in the referenced array corresponds to one instance of
        // the delegate kernel.
        // Example usage:
        //
        // TfLiteIntArray* nodes_to_replace = ...;
        // TfLiteDelegateParams* params_array;
        // int num_partitions = 0;
        // TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
        //    context, delegate, nodes_to_replace, &params_array, &num_partitions));
        // for (int idx = 0; idx < num_partitions; idx++) {
        //    const auto& partition_params = params_array[idx];
        //    ...
        // }
        //
        // NOTE: The context owns the memory referenced by partition_params_array. It
        // will be cleared with another call to PreviewDelegateParitioning, or after
        // TfLiteDelegateParams::Prepare returns.
        //
        // WARNING: This is an experimental interface that is subject to change.
        TfLiteStatus (*PreviewDelegatePartitioning)(
            struct TfLiteContext *context, const TfLiteIntArray *nodes_to_replace,
            TfLiteDelegateParams **partition_params_array, int *num_partitions);

        // Returns a TfLiteTensor struct for a given index.
        // WARNING: This is an experimental interface that is subject to change.
        // WARNING: This method may not be available on all platforms.
        TfLiteTensor *(*GetTensor)(const struct TfLiteContext *context,
                                   int tensor_idx);

        // Returns a TfLiteEvalTensor struct for a given index.
        // WARNING: This is an experimental interface that is subject to change.
        // WARNING: This method may not be available on all platforms.
        TfLiteEvalTensor *(*GetEvalTensor)(const struct TfLiteContext *context,
                                           int tensor_idx);
    } TfLiteContext;

    typedef struct TfLiteRegistration
    {
        // Initializes the op from serialized data.
        // If a built-in op:
        //   `buffer` is the op's params data (TfLiteLSTMParams*).
        //   `length` is zero.
        // If custom op:
        //   `buffer` is the op's `custom_options`.
        //   `length` is the size of the buffer.
        //
        // Returns a type-punned (i.e. void*) opaque data (e.g. a primitive pointer
        // or an instance of a struct).
        //
        // The returned pointer will be stored with the node in the `user_data` field,
        // accessible within prepare and invoke functions below.
        // NOTE: if the data is already in the desired format, simply implement this
        // function to return `nullptr` and implement the free function to be a no-op.
        void *(*init)(TfLiteContext *context, const char *buffer, size_t length);

        // The pointer `buffer` is the data previously returned by an init invocation.
        void (*free)(TfLiteContext *context, void *buffer);

        // prepare is called when the inputs this node depends on have been resized.
        // context->ResizeTensor() can be called to request output tensors to be
        // resized.
        //
        // Returns kTfLiteOk on success.
        TfLiteStatus (*prepare)(TfLiteContext *context, TfLiteNode *node);

        // Execute the node (should read node->inputs and output to node->outputs).
        // Returns kTfLiteOk on success.
        TfLiteStatus (*invoke)(TfLiteContext *context, TfLiteNode *node);

        // profiling_string is called during summarization of profiling information
        // in order to group executions together. Providing a value here will cause a
        // given op to appear multiple times is the profiling report. This is
        // particularly useful for custom ops that can perform significantly
        // different calculations depending on their `user-data`.
        const char *(*profiling_string)(const TfLiteContext *context,
                                        const TfLiteNode *node);

        // Builtin codes. If this kernel refers to a builtin this is the code
        // of the builtin. This is so we can do marshaling to other frameworks like
        // NN API.
        // Note: It is the responsibility of the registration binder to set this
        // properly.
        int32_t builtin_code;

        // Custom op name. If the op is a builtin, this will be null.
        // Note: It is the responsibility of the registration binder to set this
        // properly.
        // WARNING: This is an experimental interface that is subject to change.
        const char *custom_name;

        // The version of the op.
        // Note: It is the responsibility of the registration binder to set this
        // properly.
        int version;
    } TfLiteRegistration;

    // The flags used in `TfLiteDelegate`. Note that this is a bitmask, so the
    // values should be 1, 2, 4, 8, ...etc.
    typedef enum TfLiteDelegateFlags
    {
        kTfLiteDelegateFlagsNone = 0,
        // The flag is set if the delegate can handle dynamic sized tensors.
        // For example, the output shape of a `Resize` op with non-constant shape
        // can only be inferred when the op is invoked.
        // In this case, the Delegate is responsible for calling
        // `SetTensorToDynamic` to mark the tensor as a dynamic tensor, and calling
        // `ResizeTensor` when invoking the op.
        //
        // If the delegate isn't capable to handle dynamic tensors, this flag need
        // to be set to false.
        kTfLiteDelegateFlagsAllowDynamicTensors = 1,

        // This flag can be used by delegates (that allow dynamic tensors) to ensure
        // applicable tensor shapes are automatically propagated in the case of tensor
        // resizing.
        // This means that non-dynamic (allocation_type != kTfLiteDynamic) I/O tensors
        // of a delegate kernel will have correct shapes before its Prepare() method
        // is called. The runtime leverages TFLite builtin ops in the original
        // execution plan to propagate shapes.
        //
        // A few points to note:
        // 1. This requires kTfLiteDelegateFlagsAllowDynamicTensors. If that flag is
        // false, this one is redundant since the delegate kernels are re-initialized
        // every time tensors are resized.
        // 2. Enabling this flag adds some overhead to AllocateTensors(), since extra
        // work is required to prepare the original execution plan.
        // 3. This flag requires that the original execution plan only have ops with
        // valid registrations (and not 'dummy' custom ops like with Flex).
        // WARNING: This feature is experimental and subject to change.
        kTfLiteDelegateFlagsRequirePropagatedShapes = 2
    } TfLiteDelegateFlags;

    // WARNING: This is an experimental interface that is subject to change.
    typedef struct TfLiteDelegate
    {
        // Data that delegate needs to identify itself. This data is owned by the
        // delegate. The delegate is owned in the user code, so the delegate is
        // responsible for doing this when it is destroyed.
        void *data_;

        // Invoked by ModifyGraphWithDelegate. This prepare is called, giving the
        // delegate a view of the current graph through TfLiteContext*. It typically
        // will look at the nodes and call ReplaceNodeSubsetsWithDelegateKernels()
        // to ask the TensorFlow lite runtime to create macro-nodes to represent
        // delegated subgraphs of the original graph.
        TfLiteStatus (*Prepare)(TfLiteContext *context,
                                struct TfLiteDelegate *delegate);

        // Copy the data from delegate buffer handle into raw memory of the given
        // 'tensor'. Note that the delegate is allowed to allocate the raw bytes as
        // long as it follows the rules for kTfLiteDynamic tensors, in which case this
        // cannot be null.
        TfLiteStatus (*CopyFromBufferHandle)(TfLiteContext *context,
                                             struct TfLiteDelegate *delegate,
                                             TfLiteBufferHandle buffer_handle,
                                             TfLiteTensor *tensor);

        // Copy the data from raw memory of the given 'tensor' to delegate buffer
        // handle. This can be null if the delegate doesn't use its own buffer.
        TfLiteStatus (*CopyToBufferHandle)(TfLiteContext *context,
                                           struct TfLiteDelegate *delegate,
                                           TfLiteBufferHandle buffer_handle,
                                           TfLiteTensor *tensor);

        // Free the Delegate Buffer Handle. Note: This only frees the handle, but
        // this doesn't release the underlying resource (e.g. textures). The
        // resources are either owned by application layer or the delegate.
        // This can be null if the delegate doesn't use its own buffer.
        void (*FreeBufferHandle)(TfLiteContext *context,
                                 struct TfLiteDelegate *delegate,
                                 TfLiteBufferHandle *handle);

        // Bitmask flags. See the comments in `TfLiteDelegateFlags`.
        int64_t flags;
    } TfLiteDelegate;

    // Build a 'null' delegate, with all the fields properly set to their default
    // values.
    TfLiteDelegate TfLiteDelegateCreate();

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
#endif // TENSORFLOW_LITE_C_COMMON_H_

// End of common.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of tensor_ctypes.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_



namespace tflite
{

    template <typename T>
    inline T *GetTensorData(TfLiteTensor *tensor)
    {
        return tensor != nullptr ? reinterpret_cast<T *>(tensor->data.raw) : nullptr;
    }

    template <typename T>
    inline const T *GetTensorData(const TfLiteTensor *tensor)
    {
        return tensor != nullptr ? reinterpret_cast<const T *>(tensor->data.raw)
                                 : nullptr;
    }

    inline RuntimeShape GetTensorShape(const TfLiteTensor *tensor)
    {
        if (tensor == nullptr)
        {
            return RuntimeShape();
        }

        TfLiteIntArray *dims = tensor->dims;
        const int dims_size = dims->size;
        const int32_t *dims_data = reinterpret_cast<const int32_t *>(dims->data);
        return RuntimeShape(dims_size, dims_data);
    }

} // namespace tflite

#endif // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_

// End of tensor_ctypes.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of trained_model_compiled.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef trained_model_GEN_H
#define trained_model_GEN_H

// Sets up the model with init and prepare steps.
TfLiteStatus trained_model_init(void *(*alloc_fnc)(size_t, size_t));
// Returns the input tensor with the given index.
TfLiteTensor *trained_model_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *trained_model_output(int index);
// Runs inference for the model.
TfLiteStatus trained_model_invoke();
// Frees memory allocated
TfLiteStatus trained_model_reset(void (*free)(void *ptr));

// Returns the number of input tensors.
inline size_t trained_model_inputs()
{
    return 1;
}
// Returns the number of output tensors.
inline size_t trained_model_outputs()
{
    return 1;
}

inline void *trained_model_input_ptr(int index)
{
    return trained_model_input(index)->data.data;
}
inline size_t trained_model_input_size(int index)
{
    return trained_model_input(index)->bytes;
}
inline int trained_model_input_dims_len(int index)
{
    return trained_model_input(index)->dims->data[0];
}
inline int *trained_model_input_dims(int index)
{
    return &trained_model_input(index)->dims->data[1];
}

inline void *trained_model_output_ptr(int index)
{
    return trained_model_output(index)->data.data;
}
inline size_t trained_model_output_size(int index)
{
    return trained_model_output(index)->bytes;
}
inline int trained_model_output_dims_len(int index)
{
    return trained_model_output(index)->dims->data[0];
}
inline int *trained_model_output_dims(int index)
{
    return &trained_model_output(index)->dims->data[1];
}

#endif

// End of mtrained_model_compiled.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_aligned_malloc.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EDGE_IMPULSE_ALIGNED_MALLOC_H_
#define _EDGE_IMPULSE_ALIGNED_MALLOC_H_

#ifdef __cplusplus
namespace
{
#endif // __cplusplus

/**
 * Based on https://github.com/embeddedartistry/embedded-resources/blob/master/examples/c/malloc_aligned.c
 */

/**
 * Simple macro for making sure memory addresses are aligned
 * to the nearest power of two
 */
#ifndef align_up
#define align_up(num, align) \
    (((num) + ((align)-1)) & ~((align)-1))
#endif

    // Number of bytes we're using for storing the aligned pointer offset
    typedef uint16_t offset_t;
#define PTR_OFFSET_SZ sizeof(offset_t)

    /**
     * aligned_malloc takes in the requested alignment and size
     *	We will call malloc with extra bytes for our header and the offset
     *	required to guarantee the desired alignment.
     */
    __attribute__((unused)) void *ei_aligned_calloc(size_t align, size_t size)
    {
        void *ptr = NULL;

        // We want it to be a power of two since align_up operates on powers of two
        assert((align & (align - 1)) == 0);

        if (align && size)
        {
            /*
             * We know we have to fit an offset value
             * We also allocate extra bytes to ensure we can meet the alignment
             */
            uint32_t hdr_size = PTR_OFFSET_SZ + (align - 1);
            void *p = ei_calloc(size + hdr_size, 1);

            if (p)
            {
                /*
                 * Add the offset size to malloc's pointer (we will always store that)
                 * Then align the resulting value to the arget alignment
                 */
                ptr = (void *)align_up(((uintptr_t)p + PTR_OFFSET_SZ), align);

                // Calculate the offset and store it behind our aligned pointer
                *((offset_t *)ptr - 1) = (offset_t)((uintptr_t)ptr - (uintptr_t)p);

            } // else NULL, could not malloc
        }     // else NULL, invalid arguments

        return ptr;
    }

    /**
     * aligned_free works like free(), but we work backwards from the returned
     * pointer to find the correct offset and pointer location to return to free()
     * Note that it is VERY BAD to call free() on an aligned_malloc() pointer.
     */
    __attribute__((unused)) void ei_aligned_free(void *ptr)
    {
        assert(ptr);

        /*
         * Walk backwards from the passed-in pointer to get the pointer offset
         * We convert to an offset_t pointer and rely on pointer math to get the data
         */
        offset_t offset = *((offset_t *)ptr - 1);

        /*
         * Once we have the offset, we can get our original pointer and call free
         */
        void *p = (void *)((uint8_t *)ptr - offset);
        ei_free(p);
    }

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _EDGE_IMPULSE_ALIGNED_MALLOC_H_

// End of ei_aligned_malloc.h-----------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of edge-impulse-sdk/porting/ei_classifier_porting.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_PORTING_H_
#define _EI_CLASSIFIER_PORTING_H_

#if defined(__cplusplus) && EI_C_LINKAGE == 1
extern "C"
{
#endif // defined(__cplusplus)

    typedef enum
    {
        EI_IMPULSE_OK = 0,
        EI_IMPULSE_ERROR_SHAPES_DONT_MATCH = -1,
        EI_IMPULSE_CANCELED = -2,
        EI_IMPULSE_TFLITE_ERROR = -3,
        EI_IMPULSE_DSP_ERROR = -5,
        EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED = -6,
        EI_IMPULSE_CUBEAI_ERROR = -7,
        EI_IMPULSE_ALLOC_FAILED = -8,
        EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES = -9,
        EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE = -10,
        EI_IMPULSE_OUT_OF_MEMORY = -11,
        EI_IMPULSE_NOT_SUPPORTED_WITH_I16 = -12,
        EI_IMPULSE_INPUT_TENSOR_WAS_NULL = -13,
        EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL = -14,
        EI_IMPULSE_SCORE_TENSOR_WAS_NULL = -15,
        EI_IMPULSE_LABEL_TENSOR_WAS_NULL = -16,
        EI_IMPULSE_TENSORRT_INIT_FAILED = -17
    } EI_IMPULSE_ERROR;

    /**
     * Cancelable sleep, can be triggered with signal from other thread
     */
    EI_IMPULSE_ERROR ei_sleep(int32_t time_ms);

    /**
     * Check if the sampler thread was canceled, use this in conjunction with
     * the same signaling mechanism as ei_sleep
     */
    EI_IMPULSE_ERROR ei_run_impulse_check_canceled();

    /**
     * Read the millisecond timer
     */
    uint64_t ei_read_timer_ms();

    /**
     * Read the microsecond timer
     */
    uint64_t ei_read_timer_us();

    /**
     * Set Serial baudrate
     */
    void ei_serial_set_baudrate(int baudrate);

    /**
     * @brief      Connect to putchar of target
     *
     * @param[in]  c The chararater
     */
    void ei_putchar(char c);

    /**
     * Print wrapper around printf()
     * This is used internally to print debug information.
     */
    __attribute__((format(printf, 1, 2))) void ei_printf(const char *format, ...);

    /**
     * Override this function if your target cannot properly print floating points
     * If not overriden, this will be sent through `ei_printf()`.
     */
    void ei_printf_float(float f);

    /**
     * Wrapper around malloc
     */
    void *ei_malloc(size_t size);

    /**
     * Wrapper around calloc
     */
    void *ei_calloc(size_t nitems, size_t size);

    /**
     * Wrapper around free
     */
    void ei_free(void *ptr);

#if defined(__cplusplus) && EI_C_LINKAGE == 1
}
#endif // defined(__cplusplus) && EI_C_LINKAGE == 1

// Load porting layer depending on target
#ifndef EI_PORTING_ARDUINO
#ifdef ARDUINO
#define EI_PORTING_ARDUINO 1
#else
#define EI_PORTING_ARDUINO 0
#endif
#endif

#ifndef EI_PORTING_ECM3532
#ifdef ECM3532
#define EI_PORTING_ECM3532 1
#else
#define EI_PORTING_ECM3532 0
#endif
#endif

#ifndef EI_PORTING_ESPRESSIF
#ifdef CONFIG_IDF_TARGET_ESP32
#define EI_PORTING_ESPRESSIF 1
#else
#define EI_PORTING_ESPRESSIF 0
#endif
#endif

#ifndef EI_PORTING_MBED
#ifdef __MBED__
#define EI_PORTING_MBED 1
#else
#define EI_PORTING_MBED 0
#endif
#endif

#ifndef EI_PORTING_POSIX
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#define EI_PORTING_POSIX 1
#else
#define EI_PORTING_POSIX 0
#endif
#endif

#ifndef EI_PORTING_SILABS
#if defined(EFR32MG12P332F1024GL125)
#define EI_PORTING_SILABS 1
#else
#define EI_PORTING_SILABS 0
#endif
#endif

#ifndef EI_PORTING_RASPBERRY
#ifdef PICO_BOARD
#define EI_PORTING_RASPBERRY 1
#else
#define EI_PORTING_RASPBERRY 0
#endif
#endif

#ifndef EI_PORTING_ZEPHYR
#if defined(__ZEPHYR__)
#define EI_PORTING_ZEPHYR 1
#else
#define EI_PORTING_ZEPHYR 0
#endif
#endif

#ifndef EI_PORTING_STM32_CUBEAI
#if defined(USE_HAL_DRIVER) && !defined(__MBED__) && EI_PORTING_ZEPHYR == 0
#define EI_PORTING_STM32_CUBEAI 1
#else
#define EI_PORTING_STM32_CUBEAI 0
#endif
#endif

#ifndef EI_PORTING_HIMAX
#ifdef CPU_ARC
#define EI_PORTING_HIMAX 1
#else
#define EI_PORTING_HIMAX 0
#endif
#endif

#ifndef EI_PORTING_MINGW32
#ifdef __MINGW32__
#define EI_PORTING_MINGW32 1
#else
#define EI_PORTING_MINGW32 0
#endif
#endif
// End load porting layer depending on target

#endif // _EI_CLASSIFIER_PORTING_H_

// end of edge-impulse-sdk/porting/ei_classifier_porting.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_run_dsp.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EDGE_IMPULSE_RUN_DSP_H_
#define _EDGE_IMPULSE_RUN_DSP_H_



#if defined(__cplusplus) && EI_C_LINKAGE == 1
extern "C"
{
    extern void ei_printf(const char *format, ...);
}
#else
extern void ei_printf(const char *format, ...);
#endif

#ifdef __cplusplus
namespace
{
#endif // __cplusplus

    using namespace ei;

#if defined(EI_DSP_IMAGE_BUFFER_STATIC_SIZE)
// float ei_dsp_image_buffer[EI_DSP_IMAGE_BUFFER_STATIC_SIZE];
#endif

    // this is the frame we work on... allocate it statically so we share between invocations
    static float *ei_dsp_cont_current_frame = nullptr;
    static size_t ei_dsp_cont_current_frame_size = 0;
    static int ei_dsp_cont_current_frame_ix = 0;

    __attribute__((unused)) int extract_spectral_analysis_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float frequency)
    {
        ei_dsp_config_spectral_analysis_t config = *((ei_dsp_config_spectral_analysis_t *)config_ptr);

        int ret;

        const float sampling_freq = frequency;

        // input matrix from the raw signal
        matrix_t input_matrix(signal->total_length / config.axes, config.axes);
        if (!input_matrix.buffer)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        signal->get_data(0, signal->total_length, input_matrix.buffer);

        // scale the signal
        ret = numpy::scale(&input_matrix, config.scale_axes);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Failed to scale signal (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        // transpose the matrix so we have one row per axis (nifty!)
        ret = numpy::transpose(&input_matrix);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Failed to transpose matrix (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        // the spectral edges that we want to calculate
        matrix_t edges_matrix_in(64, 1);
        size_t edge_matrix_ix = 0;

        char spectral_str[128] = {0};
        if (strlen(config.spectral_power_edges) > sizeof(spectral_str) - 1)
        {
            EIDSP_ERR(EIDSP_PARAMETER_INVALID);
        }
        memcpy(spectral_str, config.spectral_power_edges, strlen(config.spectral_power_edges));

        // convert spectral_power_edges (string) into float array
        char *spectral_ptr = spectral_str;
        while (spectral_ptr != NULL)
        {
            while ((*spectral_ptr) == ' ')
            {
                spectral_ptr++;
            }

            edges_matrix_in.buffer[edge_matrix_ix++] = atof(spectral_ptr);

            // find next (spectral) delimiter (or '\0' character)
            while ((*spectral_ptr != ','))
            {
                spectral_ptr++;
                if (*spectral_ptr == '\0')
                    break;
            }

            if (*spectral_ptr == '\0')
            {
                spectral_ptr = NULL;
            }
            else
            {
                spectral_ptr++;
            }
        }
        edges_matrix_in.rows = edge_matrix_ix;

        // calculate how much room we need for the output matrix
        size_t output_matrix_cols = spectral::feature::calculate_spectral_buffer_size(
            true, config.spectral_peaks_count, edges_matrix_in.rows);
        // ei_printf("output_matrix_size %hux%zu\n", input_matrix.rows, output_matrix_cols);
        if (output_matrix->cols * output_matrix->rows != static_cast<uint32_t>(output_matrix_cols * config.axes))
        {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        output_matrix->cols = output_matrix_cols;
        output_matrix->rows = config.axes;

        spectral::filter_t filter_type;
        if (strcmp(config.filter_type, "low") == 0)
        {
            filter_type = spectral::filter_lowpass;
        }
        else if (strcmp(config.filter_type, "high") == 0)
        {
            filter_type = spectral::filter_highpass;
        }
        else
        {
            filter_type = spectral::filter_none;
        }

        ret = spectral::feature::spectral_analysis(output_matrix, &input_matrix,
                                                   sampling_freq, filter_type, config.filter_cutoff, config.filter_order,
                                                   config.fft_length, config.spectral_peaks_count, config.spectral_peaks_threshold, &edges_matrix_in);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Failed to calculate spectral features (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        // flatten again
        output_matrix->cols = config.axes * output_matrix_cols;
        output_matrix->rows = 1;

        return EIDSP_OK;
    }

    __attribute__((unused)) int extract_raw_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float frequency)
    {
        ei_dsp_config_raw_t config = *((ei_dsp_config_raw_t *)config_ptr);

        // input matrix from the raw signal
        matrix_t input_matrix(signal->total_length / config.axes, config.axes);
        if (!input_matrix.buffer)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }
        signal->get_data(0, signal->total_length, input_matrix.buffer);

        // scale the signal
        int ret = numpy::scale(&input_matrix, config.scale_axes);
        if (ret != EIDSP_OK)
        {
            EIDSP_ERR(ret);
        }

        // Because of rounding errors during re-sampling the output size of the block might be
        // smaller than the input of the block. Make sure we don't write outside of the bounds
        // of the array:
        // https://forum.edgeimpulse.com/t/using-custom-sensors-on-raspberry-pi-4/3506/7
        size_t els_to_copy = signal->total_length;
        if (els_to_copy > output_matrix->rows * output_matrix->cols)
        {
            els_to_copy = output_matrix->rows * output_matrix->cols;
        }

        memcpy(output_matrix->buffer, input_matrix.buffer, els_to_copy * sizeof(float));

        return EIDSP_OK;
    }

    __attribute__((unused)) int extract_flatten_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float frequency)
    {
        ei_dsp_config_flatten_t config = *((ei_dsp_config_flatten_t *)config_ptr);

        uint32_t expected_matrix_size = 0;
        if (config.average)
            expected_matrix_size += config.axes;
        if (config.minimum)
            expected_matrix_size += config.axes;
        if (config.maximum)
            expected_matrix_size += config.axes;
        if (config.rms)
            expected_matrix_size += config.axes;
        if (config.stdev)
            expected_matrix_size += config.axes;
        if (config.skewness)
            expected_matrix_size += config.axes;
        if (config.kurtosis)
            expected_matrix_size += config.axes;

        if (output_matrix->rows * output_matrix->cols != expected_matrix_size)
        {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        int ret;

        // input matrix from the raw signal
        matrix_t input_matrix(signal->total_length / config.axes, config.axes);
        if (!input_matrix.buffer)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }
        signal->get_data(0, signal->total_length, input_matrix.buffer);

        // scale the signal
        ret = numpy::scale(&input_matrix, config.scale_axes);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Failed to scale signal (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        // transpose the matrix so we have one row per axis (nifty!)
        ret = numpy::transpose(&input_matrix);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Failed to transpose matrix (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        size_t out_matrix_ix = 0;

        for (size_t row = 0; row < input_matrix.rows; row++)
        {
            matrix_t row_matrix(1, input_matrix.cols, input_matrix.buffer + (row * input_matrix.cols));

            if (config.average)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::mean(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }

            if (config.minimum)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::min(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }

            if (config.maximum)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::max(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }

            if (config.rms)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::rms(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }

            if (config.stdev)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::stdev(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }

            if (config.skewness)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::skew(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }

            if (config.kurtosis)
            {
                float fbuffer;
                matrix_t out_matrix(1, 1, &fbuffer);
                numpy::kurtosis(&row_matrix, &out_matrix);
                output_matrix->buffer[out_matrix_ix++] = out_matrix.buffer[0];
            }
        }

        // flatten again
        output_matrix->cols = output_matrix->rows * output_matrix->cols;
        output_matrix->rows = 1;

        return EIDSP_OK;
    }

    static class speechpy::processing::preemphasis *preemphasis;
    static int preemphasized_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
    {
        return preemphasis->get_data(offset, length, out_ptr);
    }

    __attribute__((unused)) int extract_mfcc_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency)
    {
        ei_dsp_config_mfcc_t config = *((ei_dsp_config_mfcc_t *)config_ptr);

        if (config.axes != 1)
        {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if ((config.implementation_version == 0) || (config.implementation_version > 3))
        {
            EIDSP_ERR(EIDSP_BLOCK_VERSION_INCORRECT);
        }

        if (signal->total_length == 0)
        {
            EIDSP_ERR(EIDSP_PARAMETER_INVALID);
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
        if (out_matrix_size.rows * out_matrix_size.cols > output_matrix->rows * output_matrix->cols)
        {
            ei_printf("out_matrix = %dx%d\n", (int)output_matrix->rows, (int)output_matrix->cols);
            ei_printf("calculated size = %dx%d\n", (int)out_matrix_size.rows, (int)out_matrix_size.cols);
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        output_matrix->rows = out_matrix_size.rows;
        output_matrix->cols = out_matrix_size.cols;

        // and run the MFCC extraction (using 32 rather than 40 filters here to optimize speed on embedded)
        int ret = speechpy::feature::mfcc(output_matrix, &preemphasized_audio_signal,
                                          frequency, config.frame_length, config.frame_stride, config.num_cepstral, config.num_filters, config.fft_length,
                                          config.low_frequency, config.high_frequency, true, config.implementation_version);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: MFCC failed (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        // cepstral mean and variance normalization
        ret = speechpy::processing::cmvnw(output_matrix, config.win_size, true, false);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: cmvnw failed (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        output_matrix->cols = out_matrix_size.rows * out_matrix_size.cols;
        output_matrix->rows = 1;

        return EIDSP_OK;
    }

    static int extract_mfcc_run_slice(signal_t *signal, matrix_t *output_matrix, ei_dsp_config_mfcc_t *config, const float sampling_frequency, matrix_size_t *matrix_size_out, int implementation_version)
    {
        uint32_t frequency = (uint32_t)sampling_frequency;

        int x;

        // calculate the size of the spectrogram matrix
        matrix_size_t out_matrix_size =
            speechpy::feature::calculate_mfcc_buffer_size(
                signal->total_length, frequency, config->frame_length, config->frame_stride, config->num_cepstral,
                implementation_version);

        // we roll the output matrix back so we have room at the end...
        x = numpy::roll(output_matrix->buffer, output_matrix->rows * output_matrix->cols,
                        -(out_matrix_size.rows * out_matrix_size.cols));
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        // slice in the output matrix to write to
        // the offset in the classification matrix here is always at the end
        size_t output_matrix_offset = (output_matrix->rows * output_matrix->cols) -
                                      (out_matrix_size.rows * out_matrix_size.cols);

        matrix_t output_matrix_slice(out_matrix_size.rows, out_matrix_size.cols, output_matrix->buffer + output_matrix_offset);

        // and run the MFCC extraction
        x = speechpy::feature::mfcc(&output_matrix_slice, signal,
                                    frequency, config->frame_length, config->frame_stride, config->num_cepstral, config->num_filters, config->fft_length,
                                    config->low_frequency, config->high_frequency, true, implementation_version);
        if (x != EIDSP_OK)
        {
            ei_printf("ERR: MFCC failed (%d)\n", x);
            EIDSP_ERR(x);
        }

        matrix_size_out->rows += out_matrix_size.rows;
        if (out_matrix_size.cols > 0)
        {
            matrix_size_out->cols = out_matrix_size.cols;
        }

        return EIDSP_OK;
    }

    __attribute__((unused)) int extract_mfcc_per_slice_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency, matrix_size_t *matrix_size_out)
    {
#if defined(__cplusplus) && EI_C_LINKAGE == 1
        ei_printf("ERR: Continuous audio is not supported when EI_C_LINKAGE is defined\n");
        EIDSP_ERR(EIDSP_NOT_SUPPORTED);
#else

    ei_dsp_config_mfcc_t config = *((ei_dsp_config_mfcc_t *)config_ptr);

    if (config.axes != 1)
    {
        EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
    }

    if ((config.implementation_version == 0) || (config.implementation_version > 3))
    {
        EIDSP_ERR(EIDSP_BLOCK_VERSION_INCORRECT);
    }

    if (signal->total_length == 0)
    {
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    const uint32_t frequency = static_cast<uint32_t>(sampling_frequency);

    // preemphasis class to preprocess the audio...
    class speechpy::processing::preemphasis pre(signal, config.pre_shift, config.pre_cof, false);
    preemphasis = &pre;

    signal_t preemphasized_audio_signal;
    preemphasized_audio_signal.total_length = signal->total_length;
    preemphasized_audio_signal.get_data = &preemphasized_audio_signal_get_data;

    // Go from the time (e.g. 0.25 seconds to number of frames based on freq)
    const size_t frame_length_values = frequency * config.frame_length;
    const size_t frame_stride_values = frequency * config.frame_stride;
    const int frame_overlap_values = static_cast<int>(frame_length_values) - static_cast<int>(frame_stride_values);

    if (frame_overlap_values < 0)
    {
        ei_printf("ERR: frame_length (%f) cannot be lower than frame_stride (%f) for continuous classification\n",
                  config.frame_length, config.frame_stride);
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    int x;

    // have current frame, but wrong size? then free
    if (ei_dsp_cont_current_frame && ei_dsp_cont_current_frame_size != frame_length_values)
    {
        ei_free(ei_dsp_cont_current_frame);
        ei_dsp_cont_current_frame = nullptr;
    }

    int implementation_version = config.implementation_version;

    // this is the offset in the signal from which we'll work
    size_t offset_in_signal = 0;

    if (!ei_dsp_cont_current_frame)
    {
        ei_dsp_cont_current_frame = (float *)ei_calloc(frame_length_values * sizeof(float), 1);
        if (!ei_dsp_cont_current_frame)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }
        ei_dsp_cont_current_frame_size = frame_length_values;
        ei_dsp_cont_current_frame_ix = 0;
    }

    if ((frame_length_values) > preemphasized_audio_signal.total_length + ei_dsp_cont_current_frame_ix)
    {
        ei_printf("ERR: frame_length (%d) cannot be larger than signal's total length (%d) for continuous classification\n",
                  (int)frame_length_values, (int)preemphasized_audio_signal.total_length + ei_dsp_cont_current_frame_ix);
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    matrix_size_out->rows = 0;
    matrix_size_out->cols = 0;

    // for continuous use v2 stack frame calculations
    if (implementation_version == 1)
    {
        implementation_version = 2;
    }

    if (ei_dsp_cont_current_frame_ix > (int)ei_dsp_cont_current_frame_size)
    {
        ei_printf("ERR: ei_dsp_cont_current_frame_ix is larger than frame size (ix=%d size=%d)\n",
                  ei_dsp_cont_current_frame_ix, (int)ei_dsp_cont_current_frame_size);
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    // if we still have some code from previous run
    while (ei_dsp_cont_current_frame_ix > 0)
    {
        // then from the current frame we need to read `frame_length_values - ei_dsp_cont_current_frame_ix`
        // starting at offset 0
        x = preemphasized_audio_signal.get_data(0, frame_length_values - ei_dsp_cont_current_frame_ix, ei_dsp_cont_current_frame + ei_dsp_cont_current_frame_ix);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        // now ei_dsp_cont_current_frame is complete
        signal_t frame_signal;
        x = numpy::signal_from_buffer(ei_dsp_cont_current_frame, frame_length_values, &frame_signal);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        x = extract_mfcc_run_slice(&frame_signal, output_matrix, &config, sampling_frequency, matrix_size_out, implementation_version);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        // if there's overlap between frames we roll through
        if (frame_stride_values > 0)
        {
            numpy::roll(ei_dsp_cont_current_frame, frame_length_values, -frame_stride_values);
        }

        ei_dsp_cont_current_frame_ix -= frame_stride_values;
    }

    if (ei_dsp_cont_current_frame_ix < 0)
    {
        offset_in_signal = -ei_dsp_cont_current_frame_ix;
        ei_dsp_cont_current_frame_ix = 0;
    }

    if (offset_in_signal >= signal->total_length)
    {
        offset_in_signal -= signal->total_length;
        return EIDSP_OK;
    }

    // now... we need to discard part of the signal...
    SignalWithRange signal_with_range(&preemphasized_audio_signal, offset_in_signal, signal->total_length);

    signal_t *range_signal = signal_with_range.get_signal();
    size_t range_signal_orig_length = range_signal->total_length;

    // then we'll just go through normal processing of the signal:
    x = extract_mfcc_run_slice(range_signal, output_matrix, &config, sampling_frequency, matrix_size_out, implementation_version);
    if (x != EIDSP_OK)
    {
        EIDSP_ERR(x);
    }

    // Make sure v1 model are reset to the original length;
    range_signal->total_length = range_signal_orig_length;

    // update offset
    int length_of_signal_used = speechpy::processing::calculate_signal_used(range_signal->total_length, sampling_frequency,
                                                                            config.frame_length, config.frame_stride, false, implementation_version);
    offset_in_signal += length_of_signal_used;

    // see what's left?
    int bytes_left_end_of_frame = signal->total_length - offset_in_signal;
    bytes_left_end_of_frame += frame_overlap_values;

    if (bytes_left_end_of_frame > 0)
    {
        // then read that into the ei_dsp_cont_current_frame buffer
        x = preemphasized_audio_signal.get_data(
            (preemphasized_audio_signal.total_length - bytes_left_end_of_frame),
            bytes_left_end_of_frame,
            ei_dsp_cont_current_frame);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }
    }

    ei_dsp_cont_current_frame_ix = bytes_left_end_of_frame;

    preemphasis = nullptr;

    return EIDSP_OK;
#endif
    }

    __attribute__((unused)) int extract_spectrogram_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency)
    {
        ei_dsp_config_spectrogram_t config = *((ei_dsp_config_spectrogram_t *)config_ptr);

        if (config.axes != 1)
        {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (signal->total_length == 0)
        {
            EIDSP_ERR(EIDSP_PARAMETER_INVALID);
        }

        const uint32_t frequency = static_cast<uint32_t>(sampling_frequency);

        // calculate the size of the MFE matrix
        matrix_size_t out_matrix_size =
            speechpy::feature::calculate_mfe_buffer_size(
                signal->total_length, frequency, config.frame_length, config.frame_stride, config.fft_length / 2 + 1,
                config.implementation_version);
        /* Only throw size mismatch error calculated buffer doesn't fit for continuous inferencing */
        if (out_matrix_size.rows * out_matrix_size.cols > output_matrix->rows * output_matrix->cols)
        {
            ei_printf("out_matrix = %dx%d\n", (int)output_matrix->rows, (int)output_matrix->cols);
            ei_printf("calculated size = %dx%d\n", (int)out_matrix_size.rows, (int)out_matrix_size.cols);
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        output_matrix->rows = out_matrix_size.rows;
        output_matrix->cols = out_matrix_size.cols;

        // and run the MFE extraction
        EI_DSP_MATRIX(energy_matrix, output_matrix->rows, 1);
        if (!energy_matrix.buffer)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        int ret = speechpy::feature::spectrogram(output_matrix, signal,
                                                 sampling_frequency, config.frame_length, config.frame_stride, config.fft_length, config.implementation_version);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Spectrogram failed (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        if (config.implementation_version < 3)
        {
            ret = numpy::normalize(output_matrix);
            if (ret != EIDSP_OK)
            {
                EIDSP_ERR(ret);
            }
        }
        else
        {
            // normalization
            ret = speechpy::processing::spectrogram_normalization(output_matrix, config.noise_floor_db);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: normalization failed (%d)\n", ret);
                EIDSP_ERR(ret);
            }
        }

        output_matrix->cols = out_matrix_size.rows * out_matrix_size.cols;
        output_matrix->rows = 1;

        return EIDSP_OK;
    }

    static int extract_spectrogram_run_slice(signal_t *signal, matrix_t *output_matrix, ei_dsp_config_spectrogram_t *config, const float sampling_frequency, matrix_size_t *matrix_size_out)
    {
        uint32_t frequency = (uint32_t)sampling_frequency;

        int x;

        // calculate the size of the spectrogram matrix
        matrix_size_t out_matrix_size =
            speechpy::feature::calculate_mfe_buffer_size(
                signal->total_length, frequency, config->frame_length, config->frame_stride, config->fft_length / 2 + 1,
                config->implementation_version);

        // we roll the output matrix back so we have room at the end...
        x = numpy::roll(output_matrix->buffer, output_matrix->rows * output_matrix->cols,
                        -(out_matrix_size.rows * out_matrix_size.cols));
        if (x != EIDSP_OK)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(x);
        }

        // slice in the output matrix to write to
        // the offset in the classification matrix here is always at the end
        size_t output_matrix_offset = (output_matrix->rows * output_matrix->cols) -
                                      (out_matrix_size.rows * out_matrix_size.cols);

        matrix_t output_matrix_slice(out_matrix_size.rows, out_matrix_size.cols, output_matrix->buffer + output_matrix_offset);

        // and run the spectrogram extraction
        int ret = speechpy::feature::spectrogram(&output_matrix_slice, signal,
                                                 frequency, config->frame_length, config->frame_stride, config->fft_length, config->implementation_version);

        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Spectrogram failed (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        matrix_size_out->rows += out_matrix_size.rows;
        if (out_matrix_size.cols > 0)
        {
            matrix_size_out->cols = out_matrix_size.cols;
        }

        return EIDSP_OK;
    }

    __attribute__((unused)) int extract_spectrogram_per_slice_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency, matrix_size_t *matrix_size_out)
    {
#if defined(__cplusplus) && EI_C_LINKAGE == 1
        ei_printf("ERR: Continuous audio is not supported when EI_C_LINKAGE is defined\n");
        EIDSP_ERR(EIDSP_NOT_SUPPORTED);
#else

    ei_dsp_config_spectrogram_t config = *((ei_dsp_config_spectrogram_t *)config_ptr);

    static bool first_run = false;

    if (config.axes != 1)
    {
        EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
    }

    if (signal->total_length == 0)
    {
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    const uint32_t frequency = static_cast<uint32_t>(sampling_frequency);

    /* Fake an extra frame_length for stack frames calculations. There, 1 frame_length is always
    subtracted and there for never used. But skip the first slice to fit the feature_matrix
    buffer */
    if (config.implementation_version < 2)
    {

        if (first_run == true)
        {
            signal->total_length += (size_t)(config.frame_length * (float)frequency);
        }

        first_run = true;
    }

    // Go from the time (e.g. 0.25 seconds to number of frames based on freq)
    const size_t frame_length_values = frequency * config.frame_length;
    const size_t frame_stride_values = frequency * config.frame_stride;
    const int frame_overlap_values = static_cast<int>(frame_length_values) - static_cast<int>(frame_stride_values);

    if (frame_overlap_values < 0)
    {
        ei_printf("ERR: frame_length (%f) cannot be lower than frame_stride (%f) for continuous classification\n",
                  config.frame_length, config.frame_stride);
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    if (frame_length_values > signal->total_length)
    {
        ei_printf("ERR: frame_length (%d) cannot be larger than signal's total length (%d) for continuous classification\n",
                  (int)frame_length_values, (int)signal->total_length);
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    int x;

    // have current frame, but wrong size? then free
    if (ei_dsp_cont_current_frame && ei_dsp_cont_current_frame_size != frame_length_values)
    {
        ei_free(ei_dsp_cont_current_frame);
        ei_dsp_cont_current_frame = nullptr;
    }

    if (!ei_dsp_cont_current_frame)
    {
        ei_dsp_cont_current_frame = (float *)ei_calloc(frame_length_values * sizeof(float), 1);
        if (!ei_dsp_cont_current_frame)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }
        ei_dsp_cont_current_frame_size = frame_length_values;
        ei_dsp_cont_current_frame_ix = 0;
    }

    matrix_size_out->rows = 0;
    matrix_size_out->cols = 0;

    // this is the offset in the signal from which we'll work
    size_t offset_in_signal = 0;

    if (ei_dsp_cont_current_frame_ix > (int)ei_dsp_cont_current_frame_size)
    {
        ei_printf("ERR: ei_dsp_cont_current_frame_ix is larger than frame size\n");
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    // if we still have some code from previous run
    while (ei_dsp_cont_current_frame_ix > 0)
    {
        // then from the current frame we need to read `frame_length_values - ei_dsp_cont_current_frame_ix`
        // starting at offset 0
        x = signal->get_data(0, frame_length_values - ei_dsp_cont_current_frame_ix, ei_dsp_cont_current_frame + ei_dsp_cont_current_frame_ix);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        // now ei_dsp_cont_current_frame is complete
        signal_t frame_signal;
        x = numpy::signal_from_buffer(ei_dsp_cont_current_frame, frame_length_values, &frame_signal);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        x = extract_spectrogram_run_slice(&frame_signal, output_matrix, &config, sampling_frequency, matrix_size_out);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        // if there's overlap between frames we roll through
        if (frame_stride_values > 0)
        {
            numpy::roll(ei_dsp_cont_current_frame, frame_length_values, -frame_stride_values);
        }

        ei_dsp_cont_current_frame_ix -= frame_stride_values;
    }

    if (ei_dsp_cont_current_frame_ix < 0)
    {
        offset_in_signal = -ei_dsp_cont_current_frame_ix;
        ei_dsp_cont_current_frame_ix = 0;
    }

    if (offset_in_signal >= signal->total_length)
    {
        offset_in_signal -= signal->total_length;
        return EIDSP_OK;
    }

    // now... we need to discard part of the signal...
    SignalWithRange signal_with_range(signal, offset_in_signal, signal->total_length);

    signal_t *range_signal = signal_with_range.get_signal();
    size_t range_signal_orig_length = range_signal->total_length;

    // then we'll just go through normal processing of the signal:
    x = extract_spectrogram_run_slice(range_signal, output_matrix, &config, sampling_frequency, matrix_size_out);
    if (x != EIDSP_OK)
    {
        EIDSP_ERR(x);
    }

    // update offset
    int length_of_signal_used = speechpy::processing::calculate_signal_used(range_signal->total_length, sampling_frequency,
                                                                            config.frame_length, config.frame_stride, false, config.implementation_version);
    offset_in_signal += length_of_signal_used;

    // not sure why this is being manipulated...
    range_signal->total_length = range_signal_orig_length;

    // see what's left?
    int bytes_left_end_of_frame = signal->total_length - offset_in_signal;
    bytes_left_end_of_frame += frame_overlap_values;

    if (bytes_left_end_of_frame > 0)
    {
        // then read that into the ei_dsp_cont_current_frame buffer
        x = signal->get_data(
            (signal->total_length - bytes_left_end_of_frame),
            bytes_left_end_of_frame,
            ei_dsp_cont_current_frame);
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }
    }

    ei_dsp_cont_current_frame_ix = bytes_left_end_of_frame;

    if (config.implementation_version < 2)
    {
        if (first_run == true)
        {
            signal->total_length -= (size_t)(config.frame_length * (float)frequency);
        }
    }

    return EIDSP_OK;
#endif
    }

    __attribute__((unused)) int extract_mfe_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency)
    {
        ei_dsp_config_mfe_t config = *((ei_dsp_config_mfe_t *)config_ptr);

        if (config.axes != 1)
        {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (signal->total_length == 0)
        {
            EIDSP_ERR(EIDSP_PARAMETER_INVALID);
        }

        const uint32_t frequency = static_cast<uint32_t>(sampling_frequency);

        signal_t preemphasized_audio_signal;

        // before version 3 we did not have preemphasis
        if (config.implementation_version < 3)
        {
            preemphasis = nullptr;

            preemphasized_audio_signal.total_length = signal->total_length;
            preemphasized_audio_signal.get_data = signal->get_data;
        }
        else
        {
            // preemphasis class to preprocess the audio...
            class speechpy::processing::preemphasis *pre = new class speechpy::processing::preemphasis(signal, 1, 0.98f, true);
            preemphasis = pre;

            preemphasized_audio_signal.total_length = signal->total_length;
            preemphasized_audio_signal.get_data = &preemphasized_audio_signal_get_data;
        }

        // calculate the size of the MFE matrix
        matrix_size_t out_matrix_size =
            speechpy::feature::calculate_mfe_buffer_size(
                preemphasized_audio_signal.total_length, frequency, config.frame_length, config.frame_stride, config.num_filters,
                config.implementation_version);
        /* Only throw size mismatch error calculated buffer doesn't fit for continuous inferencing */
        if (out_matrix_size.rows * out_matrix_size.cols > output_matrix->rows * output_matrix->cols)
        {
            ei_printf("out_matrix = %dx%d\n", (int)output_matrix->rows, (int)output_matrix->cols);
            ei_printf("calculated size = %dx%d\n", (int)out_matrix_size.rows, (int)out_matrix_size.cols);
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        output_matrix->rows = out_matrix_size.rows;
        output_matrix->cols = out_matrix_size.cols;

        // and run the MFE extraction
        EI_DSP_MATRIX(energy_matrix, output_matrix->rows, 1);
        if (!energy_matrix.buffer)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        int ret = speechpy::feature::mfe(output_matrix, &energy_matrix, &preemphasized_audio_signal,
                                         frequency, config.frame_length, config.frame_stride, config.num_filters, config.fft_length,
                                         config.low_frequency, config.high_frequency, config.implementation_version);
        if (preemphasis)
        {
            delete preemphasis;
        }
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: MFE failed (%d)\n", ret);
            EIDSP_ERR(ret);
        }

        if (config.implementation_version < 3)
        {
            // cepstral mean and variance normalization
            ret = speechpy::processing::cmvnw(output_matrix, config.win_size, false, true);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: cmvnw failed (%d)\n", ret);
                EIDSP_ERR(ret);
            }
        }
        else
        {
            // normalization
            ret = speechpy::processing::mfe_normalization(output_matrix, config.noise_floor_db);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: normalization failed (%d)\n", ret);
                EIDSP_ERR(ret);
            }
        }

        output_matrix->cols = out_matrix_size.rows * out_matrix_size.cols;
        output_matrix->rows = 1;

        return EIDSP_OK;
    }

    static int extract_mfe_run_slice(signal_t *signal, matrix_t *output_matrix, ei_dsp_config_mfe_t *config, const float sampling_frequency, matrix_size_t *matrix_size_out)
    {
        uint32_t frequency = (uint32_t)sampling_frequency;

        int x;

        // calculate the size of the spectrogram matrix
        matrix_size_t out_matrix_size =
            speechpy::feature::calculate_mfe_buffer_size(
                signal->total_length, frequency, config->frame_length, config->frame_stride, config->num_filters,
                config->implementation_version);

        // we roll the output matrix back so we have room at the end...
        x = numpy::roll(output_matrix->buffer, output_matrix->rows * output_matrix->cols,
                        -(out_matrix_size.rows * out_matrix_size.cols));
        if (x != EIDSP_OK)
        {
            EIDSP_ERR(x);
        }

        // slice in the output matrix to write to
        // the offset in the classification matrix here is always at the end
        size_t output_matrix_offset = (output_matrix->rows * output_matrix->cols) -
                                      (out_matrix_size.rows * out_matrix_size.cols);

        matrix_t output_matrix_slice(out_matrix_size.rows, out_matrix_size.cols, output_matrix->buffer + output_matrix_offset);

        // energy matrix
        EI_DSP_MATRIX(energy_matrix, out_matrix_size.rows, 1);
        if (!energy_matrix.buffer)
        {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        // and run the MFE extraction
        x = speechpy::feature::mfe(&output_matrix_slice, &energy_matrix, signal,
                                   frequency, config->frame_length, config->frame_stride, config->num_filters, config->fft_length,
                                   config->low_frequency, config->high_frequency, config->implementation_version);
        if (x != EIDSP_OK)
        {
            ei_printf("ERR: MFE failed (%d)\n", x);
            EIDSP_ERR(x);
        }

        matrix_size_out->rows += out_matrix_size.rows;
        if (out_matrix_size.cols > 0)
        {
            matrix_size_out->cols = out_matrix_size.cols;
        }

        return EIDSP_OK;
    }

    __attribute__((unused)) int extract_mfe_per_slice_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float sampling_frequency, matrix_size_t *matrix_size_out)
    {
#if defined(__cplusplus) && EI_C_LINKAGE == 1
        ei_printf("ERR: Continuous audio is not supported when EI_C_LINKAGE is defined\n");
        EIDSP_ERR(EIDSP_NOT_SUPPORTED);
#else

    ei_dsp_config_mfe_t config = *((ei_dsp_config_mfe_t *)config_ptr);

    // signal is already the right size,
    // output matrix is not the right size, but we can start writing at offset 0 and then it's OK too

    static bool first_run = false;

    if (config.axes != 1)
    {
        EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
    }

    if (signal->total_length == 0)
    {
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    const uint32_t frequency = static_cast<uint32_t>(sampling_frequency);

    // Fake an extra frame_length for stack frames calculations. There, 1 frame_length is always
    // subtracted and there for never used. But skip the first slice to fit the feature_matrix
    // buffer
    if (config.implementation_version == 1)
    {
        if (first_run == true)
        {
            signal->total_length += (size_t)(config.frame_length * (float)frequency);
        }

        first_run = true;
    }

    // ok all setup, let's construct the signal (with preemphasis for impl version >3)
    signal_t preemphasized_audio_signal;

    // before version 3 we did not have preemphasis
    if (config.implementation_version < 3)
    {
        preemphasis = nullptr;
        preemphasized_audio_signal.total_length = signal->total_length;
        preemphasized_audio_signal.get_data = signal->get_data;
    }
    else
    {
        // preemphasis class to preprocess the audio...
        class speechpy::processing::preemphasis *pre = new class speechpy::processing::preemphasis(signal, 1, 0.98f, true);
        preemphasis = pre;
        preemphasized_audio_signal.total_length = signal->total_length;
        preemphasized_audio_signal.get_data = &preemphasized_audio_signal_get_data;
    }

    // Go from the time (e.g. 0.25 seconds to number of frames based on freq)
    const size_t frame_length_values = frequency * config.frame_length;
    const size_t frame_stride_values = frequency * config.frame_stride;
    const int frame_overlap_values = static_cast<int>(frame_length_values) - static_cast<int>(frame_stride_values);

    if (frame_overlap_values < 0)
    {
        ei_printf("ERR: frame_length (%f) cannot be lower than frame_stride (%f) for continuous classification\n",
                  config.frame_length, config.frame_stride);
        if (preemphasis)
        {
            delete preemphasis;
        }
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    if (frame_length_values > preemphasized_audio_signal.total_length)
    {
        ei_printf("ERR: frame_length (%d) cannot be larger than signal's total length (%d) for continuous classification\n",
                  (int)frame_length_values, (int)preemphasized_audio_signal.total_length);
        if (preemphasis)
        {
            delete preemphasis;
        }
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    int x;

    // have current frame, but wrong size? then free
    if (ei_dsp_cont_current_frame && ei_dsp_cont_current_frame_size != frame_length_values)
    {
        ei_free(ei_dsp_cont_current_frame);
        ei_dsp_cont_current_frame = nullptr;
    }

    if (!ei_dsp_cont_current_frame)
    {
        ei_dsp_cont_current_frame = (float *)ei_calloc(frame_length_values * sizeof(float), 1);
        if (!ei_dsp_cont_current_frame)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }
        ei_dsp_cont_current_frame_size = frame_length_values;
        ei_dsp_cont_current_frame_ix = 0;
    }

    matrix_size_out->rows = 0;
    matrix_size_out->cols = 0;

    // this is the offset in the signal from which we'll work
    size_t offset_in_signal = 0;

    if (ei_dsp_cont_current_frame_ix > (int)ei_dsp_cont_current_frame_size)
    {
        ei_printf("ERR: ei_dsp_cont_current_frame_ix is larger than frame size\n");
        if (preemphasis)
        {
            delete preemphasis;
        }
        EIDSP_ERR(EIDSP_PARAMETER_INVALID);
    }

    // if we still have some code from previous run
    while (ei_dsp_cont_current_frame_ix > 0)
    {
        // then from the current frame we need to read `frame_length_values - ei_dsp_cont_current_frame_ix`
        // starting at offset 0
        x = preemphasized_audio_signal.get_data(0, frame_length_values - ei_dsp_cont_current_frame_ix, ei_dsp_cont_current_frame + ei_dsp_cont_current_frame_ix);
        if (x != EIDSP_OK)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(x);
        }

        // now ei_dsp_cont_current_frame is complete
        signal_t frame_signal;
        x = numpy::signal_from_buffer(ei_dsp_cont_current_frame, frame_length_values, &frame_signal);
        if (x != EIDSP_OK)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(x);
        }

        x = extract_mfe_run_slice(&frame_signal, output_matrix, &config, sampling_frequency, matrix_size_out);
        if (x != EIDSP_OK)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(x);
        }

        // if there's overlap between frames we roll through
        if (frame_stride_values > 0)
        {
            numpy::roll(ei_dsp_cont_current_frame, frame_length_values, -frame_stride_values);
        }

        ei_dsp_cont_current_frame_ix -= frame_stride_values;
    }

    if (ei_dsp_cont_current_frame_ix < 0)
    {
        offset_in_signal = -ei_dsp_cont_current_frame_ix;
        ei_dsp_cont_current_frame_ix = 0;
    }

    if (offset_in_signal >= signal->total_length)
    {
        if (preemphasis)
        {
            delete preemphasis;
        }
        offset_in_signal -= signal->total_length;
        return EIDSP_OK;
    }

    // now... we need to discard part of the signal...
    SignalWithRange signal_with_range(&preemphasized_audio_signal, offset_in_signal, signal->total_length);

    signal_t *range_signal = signal_with_range.get_signal();
    size_t range_signal_orig_length = range_signal->total_length;

    // then we'll just go through normal processing of the signal:
    x = extract_mfe_run_slice(range_signal, output_matrix, &config, sampling_frequency, matrix_size_out);
    if (x != EIDSP_OK)
    {
        if (preemphasis)
        {
            delete preemphasis;
        }
        EIDSP_ERR(x);
    }

    // update offset
    int length_of_signal_used = speechpy::processing::calculate_signal_used(range_signal->total_length, sampling_frequency,
                                                                            config.frame_length, config.frame_stride, false, config.implementation_version);
    offset_in_signal += length_of_signal_used;

    // not sure why this is being manipulated...
    range_signal->total_length = range_signal_orig_length;

    // see what's left?
    int bytes_left_end_of_frame = signal->total_length - offset_in_signal;
    bytes_left_end_of_frame += frame_overlap_values;

    if (bytes_left_end_of_frame > 0)
    {
        // then read that into the ei_dsp_cont_current_frame buffer
        x = preemphasized_audio_signal.get_data(
            (preemphasized_audio_signal.total_length - bytes_left_end_of_frame),
            bytes_left_end_of_frame,
            ei_dsp_cont_current_frame);
        if (x != EIDSP_OK)
        {
            if (preemphasis)
            {
                delete preemphasis;
            }
            EIDSP_ERR(x);
        }
    }

    ei_dsp_cont_current_frame_ix = bytes_left_end_of_frame;

    if (config.implementation_version == 1)
    {
        if (first_run == true)
        {
            signal->total_length -= (size_t)(config.frame_length * (float)frequency);
        }
    }

    if (preemphasis)
    {
        delete preemphasis;
    }

    return EIDSP_OK;
#endif
    }

    __attribute__((unused)) int extract_image_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float frequency)
    {
        ei_dsp_config_image_t config = *((ei_dsp_config_image_t *)config_ptr);

        int16_t channel_count = strcmp(config.channels, "Grayscale") == 0 ? 1 : 3;

        if (output_matrix->rows * output_matrix->cols != static_cast<uint32_t>(EI_CLASSIFIER_INPUT_FRAMES * EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * channel_count))
        {
            ei_printf("out_matrix = %d items\n", static_cast<int>(output_matrix->rows * output_matrix->cols));
            ei_printf("calculated size = %d items\n", static_cast<int>(EI_CLASSIFIER_INPUT_FRAMES * EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * channel_count));
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        size_t output_ix = 0;
        const size_t page_size = 1024;

        // buffered read from the signal
        size_t bytes_left = signal->total_length;
        for (size_t ix = 0; ix < signal->total_length; ix += page_size)
        {
            size_t elements_to_read = bytes_left > page_size ? page_size : bytes_left;
            matrix_t input_matrix(elements_to_read, config.axes);

            if (!input_matrix.buffer)
            {
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
            }
            signal->get_data(ix, elements_to_read, input_matrix.buffer);

            for (size_t jx = 0; jx < elements_to_read; jx++)
            {
                uint32_t pixel = static_cast<uint32_t>(input_matrix.buffer[jx]);

                // rgb to 0..1
                float r = static_cast<float>(pixel >> 16 & 0xff) / 255.0f;
                float g = static_cast<float>(pixel >> 8 & 0xff) / 255.0f;
                float b = static_cast<float>(pixel & 0xff) / 255.0f;

                if (channel_count == 3)
                {
                    output_matrix->buffer[output_ix++] = r;
                    output_matrix->buffer[output_ix++] = g;
                    output_matrix->buffer[output_ix++] = b;
                }
                else
                {
                    // ITU-R 601-2 luma transform
                    // see: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
                    float v = (0.299f * r) + (0.587f * g) + (0.114f * b);
                    output_matrix->buffer[output_ix++] = v;
                }
            }

            bytes_left -= elements_to_read;
        }

        return EIDSP_OK;
    }

#if EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED == 1

    __attribute__((unused)) int extract_image_features_quantized(signal_t *signal, matrix_i8_t *output_matrix, void *config_ptr, const float frequency)
    {
        ei_dsp_config_image_t config = *((ei_dsp_config_image_t *)config_ptr);

        int16_t channel_count = strcmp(config.channels, "Grayscale") == 0 ? 1 : 3;

        if (output_matrix->rows * output_matrix->cols != static_cast<uint32_t>(EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * channel_count))
        {
            ei_printf("out_matrix = %d items\n", static_cast<int>(output_matrix->rows * output_matrix->cols));
            ei_printf("calculated size = %d items\n", static_cast<int>(EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * channel_count));
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        size_t output_ix = 0;

        const int32_t iRedToGray = (int32_t)(0.299f * 65536.0f);
        const int32_t iGreenToGray = (int32_t)(0.587f * 65536.0f);
        const int32_t iBlueToGray = (int32_t)(0.114f * 65536.0f);
        const size_t page_size = 1024;

        // buffered read from the signal
        size_t bytes_left = signal->total_length;
        for (size_t ix = 0; ix < signal->total_length; ix += page_size)
        {
            size_t elements_to_read = bytes_left > page_size ? page_size : bytes_left;
            matrix_t input_matrix(elements_to_read, config.axes);

            if (!input_matrix.buffer)
            {
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
            }
            signal->get_data(ix, elements_to_read, input_matrix.buffer);

            for (size_t jx = 0; jx < elements_to_read; jx++)
            {
                uint32_t pixel = static_cast<uint32_t>(input_matrix.buffer[jx]);

                if (channel_count == 3)
                {
                    // fast code path
                    if (EI_CLASSIFIER_TFLITE_INPUT_SCALE == 0.003921568859368563f && EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT == -128)
                    {
                        int32_t r = static_cast<int32_t>(pixel >> 16 & 0xff);
                        int32_t g = static_cast<int32_t>(pixel >> 8 & 0xff);
                        int32_t b = static_cast<int32_t>(pixel & 0xff);

                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(r + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(g + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(b + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                    }
                    // slow code path
                    else
                    {
                        float r = static_cast<float>(pixel >> 16 & 0xff) / 255.0f;
                        float g = static_cast<float>(pixel >> 8 & 0xff) / 255.0f;
                        float b = static_cast<float>(pixel & 0xff) / 255.0f;

                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(round(r / EI_CLASSIFIER_TFLITE_INPUT_SCALE) + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(round(g / EI_CLASSIFIER_TFLITE_INPUT_SCALE) + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(round(b / EI_CLASSIFIER_TFLITE_INPUT_SCALE) + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                    }
                }
                else
                {
                    // fast code path
                    if (EI_CLASSIFIER_TFLITE_INPUT_SCALE == 0.003921568859368563f && EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT == -128)
                    {
                        int32_t r = static_cast<int32_t>(pixel >> 16 & 0xff);
                        int32_t g = static_cast<int32_t>(pixel >> 8 & 0xff);
                        int32_t b = static_cast<int32_t>(pixel & 0xff);

                        // ITU-R 601-2 luma transform
                        // see: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
                        int32_t gray = (iRedToGray * r) + (iGreenToGray * g) + (iBlueToGray * b);
                        gray >>= 16; // scale down to int8_t
                        gray += EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT;
                        if (gray < -128)
                            gray = -128;
                        else if (gray > 127)
                            gray = 127;
                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(gray);
                    }
                    // slow code path
                    else
                    {
                        float r = static_cast<float>(pixel >> 16 & 0xff) / 255.0f;
                        float g = static_cast<float>(pixel >> 8 & 0xff) / 255.0f;
                        float b = static_cast<float>(pixel & 0xff) / 255.0f;

                        // ITU-R 601-2 luma transform
                        // see: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
                        float v = (0.299f * r) + (0.587f * g) + (0.114f * b);
                        output_matrix->buffer[output_ix++] = static_cast<int8_t>(round(v / EI_CLASSIFIER_TFLITE_INPUT_SCALE) + EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
                    }
                }
            }

            bytes_left -= elements_to_read;
        }

        return EIDSP_OK;
    }
#endif // EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED == 1

    /**
     * Clear all state regarding continuous audio. Invoke this function after continuous audio loop ends.
     */
    __attribute__((unused)) int ei_dsp_clear_continuous_audio_state()
    {
        if (ei_dsp_cont_current_frame)
        {
            ei_free(ei_dsp_cont_current_frame);
        }

        ei_dsp_cont_current_frame = nullptr;
        ei_dsp_cont_current_frame_size = 0;
        ei_dsp_cont_current_frame_ix = 0;

        return EIDSP_OK;
    }

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _EDGE_IMPULSE_RUN_DSP_H_

// End of ei_run_dsp.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_classifier_types.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EDGE_IMPULSE_RUN_CLASSIFIER_TYPES_H_
#define _EDGE_IMPULSE_RUN_CLASSIFIER_TYPES_H_

typedef struct
{
    const char *label;
    float value;
} ei_impulse_result_classification_t;

typedef struct
{
    const char *label;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    float value;
} ei_impulse_result_bounding_box_t;

typedef struct
{
    int sampling;
    int dsp;
    int classification;
    int anomaly;
    int64_t dsp_us;
    int64_t classification_us;
    int64_t anomaly_us;
} ei_impulse_result_timing_t;

typedef struct
{
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_impulse_result_bounding_box_t bounding_boxes[EI_CLASSIFIER_OBJECT_DETECTION_COUNT];
#else
    ei_impulse_result_classification_t classification[EI_CLASSIFIER_LABEL_COUNT];
#endif
    float anomaly;
    ei_impulse_result_timing_t timing;
    int32_t label_detected;
} ei_impulse_result_t;

#endif // _EDGE_IMPULSE_RUN_CLASSIFIER_TYPES_H_

// End of ei_classifier_types.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_classifier_smooth.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_SMOOTH_H_
#define _EI_CLASSIFIER_SMOOTH_H_

#if EI_CLASSIFIER_OBJECT_DETECTION != 1

typedef struct ei_classifier_smooth
{
    int *last_readings;
    size_t last_readings_size;
    uint8_t min_readings_same;
    float classifier_confidence;
    float anomaly_confidence;
    uint8_t count[EI_CLASSIFIER_LABEL_COUNT + 2] = {0};
    size_t count_size = EI_CLASSIFIER_LABEL_COUNT + 2;
} ei_classifier_smooth_t;

/**
 * Initialize a smooth structure. This is useful if you don't want to trust
 * single readings, but rather want consensus
 * (e.g. 7 / 10 readings should be the same before I draw any ML conclusions).
 * This allocates memory on the heap!
 * @param smooth Pointer to an uninitialized ei_classifier_smooth_t struct
 * @param n_readings Number of readings you want to store
 * @param min_readings_same Minimum readings that need to be the same before concluding (needs to be lower than n_readings)
 * @param classifier_confidence Minimum confidence in a class (default 0.8)
 * @param anomaly_confidence Maximum error for anomalies (default 0.3)
 */
void ei_classifier_smooth_init(ei_classifier_smooth_t *smooth, size_t n_readings,
                               uint8_t min_readings_same, float classifier_confidence = 0.8,
                               float anomaly_confidence = 0.3)
{
    smooth->last_readings = (int *)ei_malloc(n_readings * sizeof(int));
    for (size_t ix = 0; ix < n_readings; ix++)
    {
        smooth->last_readings[ix] = -1; // -1 == uncertain
    }
    smooth->last_readings_size = n_readings;
    smooth->min_readings_same = min_readings_same;
    smooth->classifier_confidence = classifier_confidence;
    smooth->anomaly_confidence = anomaly_confidence;
    smooth->count_size = EI_CLASSIFIER_LABEL_COUNT + 2;
}

/**
 * Call when a new reading comes in.
 * @param smooth Pointer to an initialized ei_classifier_smooth_t struct
 * @param result Pointer to a result structure (after calling ei_run_classifier)
 * @returns Label, either 'uncertain', 'anomaly', or a label from the result struct
 */
const char *ei_classifier_smooth_update(ei_classifier_smooth_t *smooth, ei_impulse_result_t *result)
{
    // clear out the count array
    memset(smooth->count, 0, EI_CLASSIFIER_LABEL_COUNT + 2);

    // roll through the last_readings buffer
    numpy::roll(smooth->last_readings, smooth->last_readings_size, -1);

    int reading = -1; // uncertain

    // print the predictions
    // printf("[");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
    {
        if (result->classification[ix].value >= smooth->classifier_confidence)
        {
            reading = (int)ix;
        }
    }

    smooth->last_readings[smooth->last_readings_size - 1] = reading;

    // now count last 10 readings and see what we actually see...
    for (size_t ix = 0; ix < smooth->last_readings_size; ix++)
    {
        if (smooth->last_readings[ix] >= 0)
        {
            smooth->count[smooth->last_readings[ix]]++;
        }
        else if (smooth->last_readings[ix] == -1)
        { // uncertain
            smooth->count[EI_CLASSIFIER_LABEL_COUNT]++;
        }
        else if (smooth->last_readings[ix] == -2)
        { // anomaly
            smooth->count[EI_CLASSIFIER_LABEL_COUNT + 1]++;
        }
    }

    // then loop over the count and see which is highest
    uint8_t top_result = 0;
    uint8_t top_count = 0;
    bool met_confidence_threshold = false;
    uint8_t confidence_threshold = smooth->min_readings_same; // XX% of windows should be the same
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT + 2; ix++)
    {
        if (smooth->count[ix] > top_count)
        {
            top_result = ix;
            top_count = smooth->count[ix];
        }
        if (smooth->count[ix] >= confidence_threshold)
        {
            met_confidence_threshold = true;
        }
    }

    if (met_confidence_threshold)
    {
        if (top_result == EI_CLASSIFIER_LABEL_COUNT)
        {
            return "uncertain";
        }
        else if (top_result == EI_CLASSIFIER_LABEL_COUNT + 1)
        {
            return "anomaly";
        }
        else
        {
            return result->classification[top_result].label;
        }
    }
    return "uncertain";
}

/**
 * Clear up a smooth structure
 */
void ei_classifier_smooth_free(ei_classifier_smooth_t *smooth)
{
    free(smooth->last_readings);
}

#endif // #if EI_CLASSIFIER_OBJECT_DETECTION != 1

#endif // _EI_CLASSIFIER_SMOOTH_H_

// End of ei_classifier_smooth.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_signal_with_axes.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_SIGNAL_WITH_AXES_H_
#define _EI_CLASSIFIER_SIGNAL_WITH_AXES_H_



#if !EIDSP_SIGNAL_C_FN_POINTER

using namespace ei;

class SignalWithAxes
{
public:
    SignalWithAxes(signal_t *original_signal, uint8_t *axes, size_t axes_count) : _original_signal(original_signal), _axes(axes), _axes_count(axes_count)
    {
    }

    signal_t *get_signal()
    {
        if (this->_axes_count == EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME)
        {
            return this->_original_signal;
        }

        wrapped_signal.total_length = _original_signal->total_length / EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME * _axes_count;
        wrapped_signal.get_data = mbed::callback(this, &SignalWithAxes::get_data);
        return &wrapped_signal;
    }

    int get_data(size_t offset, size_t length, float *out_ptr)
    {
        size_t offset_on_original_signal = offset / _axes_count * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
        size_t length_on_original_signal = length / _axes_count * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;

        size_t out_ptr_ix = 0;

        for (size_t ix = offset_on_original_signal; ix < offset_on_original_signal + length_on_original_signal; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME)
        {
            for (size_t axis_ix = 0; axis_ix < this->_axes_count; axis_ix++)
            {
                int r = _original_signal->get_data(ix + _axes[axis_ix], 1, &out_ptr[out_ptr_ix++]);
                if (r != 0)
                {
                    return r;
                }
            }
        }
        return 0;
    }

private:
    signal_t *_original_signal;
    uint8_t *_axes;
    size_t _axes_count;
    signal_t wrapped_signal;
};

#endif // #if !EIDSP_SIGNAL_C_FN_POINTER

#endif // _EI_CLASSIFIER_SIGNAL_WITH_AXES_H_

// End of ei_signal_with_axes.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of model-parameters/dsp_blocks.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EI_CLASSIFIER_DSP_BLOCKS_H_
#define _EI_CLASSIFIER_DSP_BLOCKS_H_

const size_t ei_dsp_blocks_size = 1;
ei_model_dsp_t ei_dsp_blocks[ei_dsp_blocks_size] = {
    {// DSP block 52
     2480,
     &extract_mfe_features,
     (void *)&ei_dsp_config_52,
     ei_dsp_config_52_axes,
     ei_dsp_config_52_axes_size}};

#endif // _EI_CLASSIFIER_DSP_BLOCKS_H_

// End of model-parameters/dsp_blocks.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_performance_calibration.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef EI_PERFORMANCE_CALIBRATION_H
#define EI_PERFORMANCE_CALIBRATION_H

/* Private const types ----------------------------------------------------- */
#define MEM_ERROR "ERR: Failed to allocate memory for performance calibration\r\n"

#define EI_PC_RET_NO_EVENT_DETECTED -1
#define EI_PC_RET_MEMORY_ERROR -2

class RecognizeEvents
{

public:
    RecognizeEvents(
        const ei_model_performance_calibration_t *config,
        uint32_t n_labels,
        uint32_t sample_length,
        float sample_interval_ms)
    {
        if ((void *)this == NULL)
        {
            ei_printf(MEM_ERROR);
            return;
        }

        this->_detection_threshold = config->detection_threshold;
        this->_suppression_flags = config->suppression_flags;
        this->_n_labels = n_labels;

        /* Determine sample length in ms */
        float sample_length_ms = (static_cast<float>(sample_length) * sample_interval_ms);

        /* Calculate number of inference runs needed for the duration window */
        this->_average_window_duration_samples =
            (config->average_window_duration_ms < static_cast<uint32_t>(sample_length_ms))
                ? 1
                : static_cast<uint32_t>(static_cast<float>(config->average_window_duration_ms) / sample_length_ms);

        /* Calculate number of inference runs for suppression */
        this->_suppression_samples = (config->suppression_ms < static_cast<uint32_t>(sample_length_ms))
                                         ? 0
                                         : static_cast<uint32_t>(static_cast<float>(config->suppression_ms) / sample_length_ms);

        /* Detection threshold should be high enough to only classifiy 1 possibly output */
        if (this->_detection_threshold <= (1.f / this->_n_labels))
        {
            ei_printf("ERR: Classifier detection threshold too low\r\n");
            return;
        }

        /* Array to store scores for all labels */
        this->_score_array = (float *)ei_malloc(
            this->_average_window_duration_samples * this->_n_labels * sizeof(float));

        if (this->_score_array == NULL)
        {
            ei_printf(MEM_ERROR);
            return;
        }

        for (int i = 0; i < this->_average_window_duration_samples * this->_n_labels; i++)
        {
            this->_score_array[i] = 0.f;
        }
        this->_score_idx = 0;

        /* Running sum for all labels */
        this->_running_sum = (float *)ei_malloc(this->_n_labels * sizeof(float));

        if (this->_running_sum != NULL)
        {
            for (int i = 0; i < this->_n_labels; i++)
            {
                this->_running_sum[i] = 0.f;
            }
        }
        else
        {
            ei_printf(MEM_ERROR);
            return;
        }

        this->_suppression_count = this->_suppression_samples;
        this->_n_scores_in_array = 0;
    }

    ~RecognizeEvents()
    {
        if (this->_score_array)
        {
            ei_free((void *)this->_score_array);
        }
        if (this->_running_sum)
        {
            ei_free((void *)this->_running_sum);
        }
    }

    int32_t trigger(ei_impulse_result_classification_t *scores)
    {
        int32_t recognized_event = EI_PC_RET_NO_EVENT_DETECTED;
        float current_top_score = 0.f;
        uint32_t current_top_index = 0;

        /* Check pointers */
        if ((void *)this == NULL || this->_score_array == NULL || this->_running_sum == NULL)
        {
            return EI_PC_RET_MEMORY_ERROR;
        }

        /* Update the score array and running sum */
        for (int i = 0; i < this->_n_labels; i++)
        {
            this->_running_sum[i] -= this->_score_array[(this->_score_idx * this->_n_labels) + i];
            this->_running_sum[i] += scores[i].value;
            this->_score_array[(this->_score_idx * this->_n_labels) + i] = scores[i].value;
        }

        if (++this->_score_idx >= this->_average_window_duration_samples)
        {
            this->_score_idx = 0;
        }

        /* Number of samples to average, increases until the buffer is full */
        if (this->_n_scores_in_array < this->_average_window_duration_samples)
        {
            this->_n_scores_in_array++;
        }

        /* Average data and place in scores & determine top score */
        for (int i = 0; i < this->_n_labels; i++)
        {
            scores[i].value = this->_running_sum[i] / this->_n_scores_in_array;

            if (scores[i].value > current_top_score)
            {
                current_top_score = scores[i].value;
                current_top_index = i;
            }
        }

        /* Check threshold, suppression */
        if (this->_suppression_samples && this->_suppression_count < this->_suppression_samples)
        {
            this->_suppression_count++;
        }
        else
        {
            if (current_top_score >= this->_detection_threshold)
            {
                recognized_event = current_top_index;

                if (this->_suppression_flags & (1 << current_top_index))
                {
                    this->_suppression_count = 0;
                }
            }
        }

        return recognized_event;
    };

    void *operator new(size_t size)
    {
        void *p = ei_malloc(size);
        return p;
    }

    void operator delete(void *p)
    {
        ei_free(p);
    }

private:
    uint32_t _average_window_duration_samples;
    float _detection_threshold;
    uint32_t _suppression_samples;
    uint32_t _suppression_count;
    uint32_t _suppression_flags;
    uint32_t _minimum_count;
    uint32_t _n_labels;
    float *_score_array;
    uint32_t _score_idx;
    float *_running_sum;
    uint32_t _n_scores_in_array;
};

#endif // EI_PERFORMANCE_CALIBRATION

// End of ei_performance_calibration.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************************************************************************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************************************************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Start of ei_run_classifier.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EDGE_IMPULSE_RUN_CLASSIFIER_H_
#define _EDGE_IMPULSE_RUN_CLASSIFIER_H_

#ifdef __cplusplus
namespace
{
#endif // __cplusplus

#define EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR (EI_CLASSIFIER_OBJECT_DETECTION && !(EI_CLASSIFIER_OBJECT_DETECTION_CONSTRAINED))

    /* Function prototypes ----------------------------------------------------- */
    extern "C" EI_IMPULSE_ERROR run_inference(ei::matrix_t *fmatrix, ei_impulse_result_t *result, bool debug);
    extern "C" EI_IMPULSE_ERROR run_classifier_image_quantized(signal_t *signal, ei_impulse_result_t *result, bool debug);
    static EI_IMPULSE_ERROR can_run_classifier_image_quantized();
    static void calc_cepstral_mean_and_var_normalization_mfcc(ei_matrix *matrix, void *config_ptr);
    static void calc_cepstral_mean_and_var_normalization_mfe(ei_matrix *matrix, void *config_ptr);
    static void calc_cepstral_mean_and_var_normalization_spectrogram(ei_matrix *matrix, void *config_ptr);

/* Private variables ------------------------------------------------------- */

// old models don't have this, add this here
#ifndef EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR
#define EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR 0
#endif // not defined EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR

    static uint64_t classifier_continuous_features_written = 0;

#if (EI_CLASSIFIER_SENSOR == EI_CLASSIFIER_SENSOR_MICROPHONE)
    static RecognizeEvents *avg_scores = NULL;
    const ei_model_performance_calibration_t ei_calibration = {
        1,
        (int32_t)(EI_CLASSIFIER_RAW_SAMPLE_COUNT / EI_CLASSIFIER_FREQUENCY) * 1000, /* Model window */
        0.8f,                                                                       /* Default threshold */
        (int32_t)(EI_CLASSIFIER_RAW_SAMPLE_COUNT / EI_CLASSIFIER_FREQUENCY) * 500,  /* Half of model window */
        0                                                                           /* Don't use flags */
    };
#endif

    /* Private functions ------------------------------------------------------- */

    /**
     * @brief      Init static vars
     */
    extern "C" void run_classifier_init(void)
    {
        classifier_continuous_features_written = 0;
        ei_dsp_clear_continuous_audio_state();

#if (EI_CLASSIFIER_SENSOR == EI_CLASSIFIER_SENSOR_MICROPHONE)
        const ei_model_performance_calibration_t *calibration = &ei_calibration;

        if (calibration != NULL)
        {
            avg_scores = new RecognizeEvents(calibration,
                                             EI_CLASSIFIER_LABEL_COUNT, EI_CLASSIFIER_SLICE_SIZE, EI_CLASSIFIER_INTERVAL_MS);
        }
#endif
    }

    extern "C" void run_classifier_deinit(void)
    {
#if (EI_CLASSIFIER_SENSOR == EI_CLASSIFIER_SENSOR_MICROPHONE)
        if ((void *)avg_scores != NULL)
        {
            delete avg_scores;
        }
#endif
    }

    /**
     * @brief      Fill the complete matrix with sample slices. From there, run inference
     *             on the matrix.
     *
     * @param      signal  Sample data
     * @param      result  Classification output
     * @param[in]  debug   Debug output enable boot
     *
     * @return     The ei impulse error.
     */
    extern "C" EI_IMPULSE_ERROR run_classifier_continuous(signal_t *signal, ei_impulse_result_t *result,
                                                          bool debug = false, bool enable_maf = true)
    {
        static ei::matrix_t static_features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);
        if (!static_features_matrix.buffer)
        {
            return EI_IMPULSE_ALLOC_FAILED;
        }

        EI_IMPULSE_ERROR ei_impulse_error = EI_IMPULSE_OK;

        uint64_t dsp_start_us = ei_read_timer_us();

        size_t out_features_index = 0;
        bool is_mfcc = false;
        bool is_mfe = false;
        bool is_spectrogram = false;

        for (size_t ix = 0; ix < ei_dsp_blocks_size; ix++)
        {
            ei_model_dsp_t block = ei_dsp_blocks[ix];

            if (out_features_index + block.n_output_features > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE)
            {
                ei_printf("ERR: Would write outside feature buffer\n");
                return EI_IMPULSE_DSP_ERROR;
            }

            ei::matrix_t fm(1, block.n_output_features,
                            static_features_matrix.buffer + out_features_index);

            int (*extract_fn_slice)(ei::signal_t * signal, ei::matrix_t * output_matrix, void *config, const float frequency, matrix_size_t *out_matrix_size);

            /* Switch to the slice version of the mfcc feature extract function */
            if (block.extract_fn == extract_mfcc_features)
            {
                extract_fn_slice = &extract_mfcc_per_slice_features;
                is_mfcc = true;
            }
            else if (block.extract_fn == extract_spectrogram_features)
            {
                extract_fn_slice = &extract_spectrogram_per_slice_features;
                is_spectrogram = true;
            }
            else if (block.extract_fn == extract_mfe_features)
            {
                extract_fn_slice = &extract_mfe_per_slice_features;
                is_mfe = true;
            }
            else
            {
                ei_printf("ERR: Unknown extract function, only MFCC, MFE and spectrogram supported\n");
                return EI_IMPULSE_DSP_ERROR;
            }

            matrix_size_t features_written;

            SignalWithAxes swa(signal, block.axes, block.axes_size);
            int ret = extract_fn_slice(swa.get_signal(), &fm, block.config, EI_CLASSIFIER_FREQUENCY, &features_written);

            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
                return EI_IMPULSE_DSP_ERROR;
            }

            if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED)
            {
                return EI_IMPULSE_CANCELED;
            }

            classifier_continuous_features_written += (features_written.rows * features_written.cols);

            out_features_index += block.n_output_features;
        }

        result->timing.dsp_us = ei_read_timer_us() - dsp_start_us;
        result->timing.dsp = (int)(result->timing.dsp_us / 1000);

        if (debug)
        {
            ei_printf("\r\nFeatures (%d ms.): ", result->timing.dsp);
            for (size_t ix = 0; ix < static_features_matrix.cols; ix++)
            {
                ei_printf_float(static_features_matrix.buffer[ix]);
                ei_printf(" ");
            }
            ei_printf("\n");
        }

        if (classifier_continuous_features_written >= EI_CLASSIFIER_NN_INPUT_FRAME_SIZE)
        {
            dsp_start_us = ei_read_timer_us();
            ei::matrix_t classify_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

            /* Create a copy of the matrix for normalization */
            for (size_t m_ix = 0; m_ix < EI_CLASSIFIER_NN_INPUT_FRAME_SIZE; m_ix++)
            {
                classify_matrix.buffer[m_ix] = static_features_matrix.buffer[m_ix];
            }

            if (is_mfcc)
            {
                calc_cepstral_mean_and_var_normalization_mfcc(&classify_matrix, ei_dsp_blocks[0].config);
            }
            else if (is_spectrogram)
            {
                calc_cepstral_mean_and_var_normalization_spectrogram(&classify_matrix, ei_dsp_blocks[0].config);
            }
            else if (is_mfe)
            {
                calc_cepstral_mean_and_var_normalization_mfe(&classify_matrix, ei_dsp_blocks[0].config);
            }
            result->timing.dsp_us += ei_read_timer_us() - dsp_start_us;
            result->timing.dsp = (int)(result->timing.dsp_us / 1000);

#if EI_CLASSIFIER_INFERENCING_ENGINE != EI_CLASSIFIER_NONE
            if (debug)
            {
                ei_printf("Running neural network...\n");
            }
#endif
            ei_impulse_error = run_inference(&classify_matrix, result, debug);

#if (EI_CLASSIFIER_SENSOR == EI_CLASSIFIER_SENSOR_MICROPHONE)
            if ((void *)avg_scores != NULL && enable_maf == true)
            {
                result->label_detected = avg_scores->trigger(result->classification);
            }
#endif
        }
        return ei_impulse_error;
    }

    /**
     * Fill the result structure from a quantized output tensor
     */
    __attribute__((unused)) static void fill_result_struct_i8(ei_impulse_result_t *result, int8_t *data, float zero_point, float scale, bool debug)
    {
        for (uint32_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
        {
            float value = static_cast<float>(data[ix] - zero_point) * scale;

            if (debug)
            {
                ei_printf("%s:\t", ei_classifier_inferencing_categories[ix]);
                ei_printf_float(value);
                ei_printf("\n");
            }
            result->classification[ix].label = ei_classifier_inferencing_categories[ix];
            result->classification[ix].value = value;
        }
    }

    /**
     * Fill the result structure from an unquantized output tensor
     */
    __attribute__((unused)) static void fill_result_struct_f32(ei_impulse_result_t *result, float *data, bool debug)
    {
        for (uint32_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
        {
            float value = data[ix];

            if (debug)
            {
                ei_printf("%s:\t", ei_classifier_inferencing_categories[ix]);
                ei_printf_float(value);
                ei_printf("\n");
            }
            result->classification[ix].label = ei_classifier_inferencing_categories[ix];
            result->classification[ix].value = value;
        }
    }

#if (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TFLITE)

    /**
     * Setup the TFLite runtime
     *
     * @param      ctx_start_us       Pointer to the start time
     * @param      input              Pointer to input tensor
     * @param      output             Pointer to output tensor
     * @param      micro_interpreter  Pointer to interpreter (for non-compiled models)
     * @param      micro_tensor_arena Pointer to the arena that will be allocated
     *
     * @return  EI_IMPULSE_OK if successful
     */
    static EI_IMPULSE_ERROR inference_tflite_setup(uint64_t *ctx_start_us, TfLiteTensor **input, TfLiteTensor **output,
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
                                                   TfLiteTensor **output_labels,
                                                   TfLiteTensor **output_scores,
#endif

                                                   ei_unique_ptr_t &p_tensor_arena)
    {

        TfLiteStatus init_status = trained_model_init(ei_aligned_calloc);
        if (init_status != kTfLiteOk)
        {
            ei_printf("Failed to allocate TFLite arena (error code %d)\n", init_status);
            return EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED;
        }

        *ctx_start_us = ei_read_timer_us();
        static bool tflite_first_run = true;
        *input = trained_model_input(EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR);
        *output = trained_model_output(EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR);
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
        *output_scores = trained_model_output(EI_CLASSIFIER_TFLITE_OUTPUT_SCORE_TENSOR);
        *output_labels = trained_model_output(EI_CLASSIFIER_TFLITE_OUTPUT_LABELS_TENSOR);
#endif // EI_CLASSIFIER_OBJECT_DETECTION

        // Assert that our quantization parameters match the model
        if (tflite_first_run)
        {
            assert((*input)->type == EI_CLASSIFIER_TFLITE_INPUT_DATATYPE);
            assert((*output)->type == EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE);
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
            assert((*output_scores)->type == EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE);
            assert((*output_labels)->type == EI_CLASSIFIER_TFLITE_OUTPUT_DATATYPE);
#endif
#if defined(EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED) || defined(EI_CLASSIFIER_TFLITE_OUTPUT_QUANTIZED)
            if (EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED)
            {
                assert((*input)->params.scale == EI_CLASSIFIER_TFLITE_INPUT_SCALE);
                assert((*input)->params.zero_point == EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT);
            }
            if (EI_CLASSIFIER_TFLITE_OUTPUT_QUANTIZED)
            {
                assert((*output)->params.scale == EI_CLASSIFIER_TFLITE_OUTPUT_SCALE);
                assert((*output)->params.zero_point == EI_CLASSIFIER_TFLITE_OUTPUT_ZEROPOINT);
            }
#endif
            tflite_first_run = false;
        }
        return EI_IMPULSE_OK;
    }

    /**
     * Run TFLite model
     *
     * @param   ctx_start_us    Start time of the setup function (see above)
     * @param   output          Output tensor
     * @param   interpreter     TFLite interpreter (non-compiled models)
     * @param   tensor_arena    Allocated arena (will be freed)
     * @param   result          Struct for results
     * @param   debug           Whether to print debug info
     *
     * @return  EI_IMPULSE_OK if successful
     */
    static EI_IMPULSE_ERROR inference_tflite_run(uint64_t ctx_start_us,
                                                 TfLiteTensor *output,
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
                                                 TfLiteTensor *labels_tensor,
                                                 TfLiteTensor *scores_tensor,
#endif
                                                 uint8_t *tensor_arena,
                                                 ei_impulse_result_t *result,
                                                 bool debug)
    {

        if (trained_model_invoke() != kTfLiteOk)
        {
            return EI_IMPULSE_TFLITE_ERROR;
        }

        uint64_t ctx_end_us = ei_read_timer_us();

        result->timing.classification_us = ctx_end_us - ctx_start_us;
        result->timing.classification = (int)(result->timing.classification_us / 1000);

        // Read the predicted y value from the model's output tensor
        if (debug)
        {
            ei_printf("Predictions (time: %d ms.):\n", result->timing.classification);
        }

        bool int8_output = output->type == TfLiteType::kTfLiteInt8;
        if (int8_output)
        {
            fill_result_struct_i8(result, output->data.int8, output->params.zero_point, output->params.scale, debug);
        }
        else
        {
            fill_result_struct_f32(result, output->data.f, debug);
        }

#if (EI_CLASSIFIER_COMPILED == 1)
        trained_model_reset(ei_aligned_free);
#endif

        if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED)
        {
            return EI_IMPULSE_CANCELED;
        }

        return EI_IMPULSE_OK;
    }
#endif // (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TFLITE)

    /**
     * @brief      Do inferencing over the processed feature matrix
     *
     * @param      fmatrix  Processed matrix
     * @param      result   Output classifier results
     * @param[in]  debug    Debug output enable
     *
     * @return     The ei impulse error.
     */
    extern "C" EI_IMPULSE_ERROR run_inference(
        ei::matrix_t *fmatrix,
        ei_impulse_result_t *result,
        bool debug = false)
    {

        {
            TfLiteTensor *input;
            TfLiteTensor *output;
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
            TfLiteTensor *output_scores;
            TfLiteTensor *output_labels;
#endif
            uint64_t ctx_start_us = ei_read_timer_us();
            ei_unique_ptr_t p_tensor_arena(nullptr, ei_aligned_free);

            EI_IMPULSE_ERROR init_res = inference_tflite_setup(&ctx_start_us, &input, &output,
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
                                                               &output_labels,
                                                               &output_scores,
#endif
                                                               p_tensor_arena);

            if (init_res != EI_IMPULSE_OK)
            {
                return init_res;
            }

            uint8_t *tensor_arena = static_cast<uint8_t *>(p_tensor_arena.get());

            // Place our calculated x value in the model's input tensor

            bool int8_input = input->type == TfLiteType::kTfLiteInt8;
            for (size_t ix = 0; ix < fmatrix->rows * fmatrix->cols; ix++)
            {
                // Quantize the input if it is int8
                if (int8_input)
                {
                    input->data.int8[ix] = static_cast<int8_t>(round(fmatrix->buffer[ix] / input->params.scale) + input->params.zero_point);
                    // printf("float %ld : %d\r\n", ix, input->data.int8[ix]);
                }
                else
                {
                    input->data.f[ix] = fmatrix->buffer[ix];
                }
            }

            EI_IMPULSE_ERROR run_res = inference_tflite_run(ctx_start_us, output,
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
                                                            output_labels,
                                                            output_scores,
#endif
                                                            tensor_arena, result, debug);

            result->timing.classification_us = ei_read_timer_us() - ctx_start_us;

            if (run_res != EI_IMPULSE_OK)
            {
                return run_res;
            }
        }

        if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED)
        {
            return EI_IMPULSE_CANCELED;
        }

        return EI_IMPULSE_OK;
    }

    /**
     * Run the classifier over a raw features array
     * @param raw_features Raw features array
     * @param raw_features_size Size of the features array
     * @param result Object to store the results in
     * @param debug Whether to show debug messages (default: false)
     */
    extern "C" EI_IMPULSE_ERROR run_classifier(
        signal_t *signal,
        ei_impulse_result_t *result,
        bool debug = false)
    {
#if EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED == 1 && (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TFLITE || EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TENSAIFLOW)

        // Shortcut for quantized image models
        if (can_run_classifier_image_quantized() == EI_IMPULSE_OK)
        {
            return run_classifier_image_quantized(signal, result, debug);
        }
#endif

        memset(result, 0, sizeof(ei_impulse_result_t));

        ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

        uint64_t dsp_start_us = ei_read_timer_us();

        size_t out_features_index = 0;

        for (size_t ix = 0; ix < ei_dsp_blocks_size; ix++)
        {
            ei_model_dsp_t block = ei_dsp_blocks[ix];

            if (out_features_index + block.n_output_features > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE)
            {
                ei_printf("ERR: Would write outside feature buffer\n");
                return EI_IMPULSE_DSP_ERROR;
            }

            ei::matrix_t fm(1, block.n_output_features, features_matrix.buffer + out_features_index);
            SignalWithAxes swa(signal, block.axes, block.axes_size);
            int ret = block.extract_fn(swa.get_signal(), &fm, block.config, EI_CLASSIFIER_FREQUENCY);

            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
                return EI_IMPULSE_DSP_ERROR;
            }

            if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED)
            {
                return EI_IMPULSE_CANCELED;
            }

            out_features_index += block.n_output_features;
        }

        result->timing.dsp_us = ei_read_timer_us() - dsp_start_us;
        result->timing.dsp = (int)(result->timing.dsp_us / 1000);

        if (debug)
        {
            ei_printf("Features (%d ms.): ", result->timing.dsp);
            for (size_t ix = 0; ix < features_matrix.cols; ix++)
            {
                ei_printf_float(features_matrix.buffer[ix]);
                ei_printf(" ");
            }
            ei_printf("\n");
        }

#if EI_CLASSIFIER_INFERENCING_ENGINE != EI_CLASSIFIER_NONE
        if (debug)
        {
            ei_printf("Running neural network...\n");
        }
#endif

        return run_inference(&features_matrix, result, debug);
    }

    /**
     * @brief      Calculates the cepstral mean and variable normalization.
     *
     * @param      matrix      Source and destination matrix
     * @param      config_ptr  ei_dsp_config_mfcc_t struct pointer
     */
    static void calc_cepstral_mean_and_var_normalization_mfcc(ei_matrix *matrix, void *config_ptr)
    {
        ei_dsp_config_mfcc_t *config = (ei_dsp_config_mfcc_t *)config_ptr;

        uint32_t original_matrix_size = matrix->rows * matrix->cols;

        /* Modify rows and colums ration for matrix normalization */
        matrix->rows = original_matrix_size / config->num_cepstral;
        matrix->cols = config->num_cepstral;

        // cepstral mean and variance normalization
        int ret = speechpy::processing::cmvnw(matrix, config->win_size, true, false);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: cmvnw failed (%d)\n", ret);
            return;
        }

        /* Reset rows and columns ratio */
        matrix->rows = 1;
        matrix->cols = original_matrix_size;
    }

    /**
     * @brief      Calculates the cepstral mean and variable normalization.
     *
     * @param      matrix      Source and destination matrix
     * @param      config_ptr  ei_dsp_config_mfe_t struct pointer
     */
    static void calc_cepstral_mean_and_var_normalization_mfe(ei_matrix *matrix, void *config_ptr)
    {
        ei_dsp_config_mfe_t *config = (ei_dsp_config_mfe_t *)config_ptr;

        uint32_t original_matrix_size = matrix->rows * matrix->cols;

        /* Modify rows and colums ration for matrix normalization */
        matrix->rows = (original_matrix_size) / config->num_filters;
        matrix->cols = config->num_filters;

        if (config->implementation_version < 3)
        {
            // cepstral mean and variance normalization
            int ret = speechpy::processing::cmvnw(matrix, config->win_size, false, true);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: cmvnw failed (%d)\n", ret);
                return;
            }
        }
        else
        {
            // normalization
            int ret = speechpy::processing::mfe_normalization(matrix, config->noise_floor_db);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: normalization failed (%d)\n", ret);
                return;
            }
        }

        /* Reset rows and columns ratio */
        matrix->rows = 1;
        matrix->cols = (original_matrix_size);
    }

    /**
     * @brief      Calculates the cepstral mean and variable normalization.
     *
     * @param      matrix      Source and destination matrix
     * @param      config_ptr  ei_dsp_config_spectrogram_t struct pointer
     */
    static void calc_cepstral_mean_and_var_normalization_spectrogram(ei_matrix *matrix, void *config_ptr)
    {
        ei_dsp_config_spectrogram_t *config = (ei_dsp_config_spectrogram_t *)config_ptr;

        uint32_t original_matrix_size = matrix->rows * matrix->cols;

        /* Modify rows and colums ration for matrix normalization */
        matrix->cols = config->fft_length / 2 + 1;
        matrix->rows = (original_matrix_size) / matrix->cols;

        if (config->implementation_version < 3)
        {
            int ret = numpy::normalize(matrix);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: normalization failed (%d)\n", ret);
                return;
            }
        }
        else
        {
            // normalization
            int ret = speechpy::processing::spectrogram_normalization(matrix, config->noise_floor_db);
            if (ret != EIDSP_OK)
            {
                ei_printf("ERR: normalization failed (%d)\n", ret);
                return;
            }
        }

        /* Reset rows and columns ratio */
        matrix->rows = 1;
        matrix->cols = (original_matrix_size);
    }

    /**
     * Check if the current impulse could be used by 'run_classifier_image_quantized'
     */
    __attribute__((unused)) static EI_IMPULSE_ERROR can_run_classifier_image_quantized()
    {

        // And if we have one DSP block which operates on images...
        if (ei_dsp_blocks_size != 1 || ei_dsp_blocks[0].extract_fn != extract_image_features)
        {
            return EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
        }

        return EI_IMPULSE_OK;
    }

#if EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED == 1 && (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TFLITE || EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TENSAIFLOW)
    /**
     * Special function to run the classifier on images, only works on TFLite models (either interpreter or EON or for tensaiflow)
     * that allocates a lot less memory by quantizing in place. This only works if 'can_run_classifier_image_quantized'
     * returns EI_IMPULSE_OK.
     */
    extern "C" EI_IMPULSE_ERROR run_classifier_image_quantized(
        signal_t *signal,
        ei_impulse_result_t *result,
        bool debug = false)
    {
        EI_IMPULSE_ERROR verify_res = can_run_classifier_image_quantized();
        if (verify_res != EI_IMPULSE_OK)
        {
            return verify_res;
        }

        memset(result, 0, sizeof(ei_impulse_result_t));

        uint64_t ctx_start_us;
        TfLiteTensor *input;
        TfLiteTensor *output;
        ei_unique_ptr_t p_tensor_arena(nullptr, ei_aligned_free);
        EI_IMPULSE_ERROR init_res = inference_tflite_setup(&ctx_start_us, &input, &output,
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
                                                           &output_labels,
                                                           &output_scores,
#endif
                                                           p_tensor_arena);

        if (init_res != EI_IMPULSE_OK)
        {
            return init_res;
        }

        if (input->type != TfLiteType::kTfLiteInt8)
        {
            return EI_IMPULSE_ONLY_SUPPORTED_FOR_IMAGES;
        }

        uint64_t dsp_start_us = ei_read_timer_us();

        // features matrix maps around the input tensor to not allocate any memory
        ei::matrix_i8_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE, input->data.int8);

        // run DSP process and quantize automatically
        int ret = extract_image_features_quantized(signal, &features_matrix, ei_dsp_blocks[0].config, EI_CLASSIFIER_FREQUENCY);
        if (ret != EIDSP_OK)
        {
            ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
            return EI_IMPULSE_DSP_ERROR;
        }

        if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED)
        {
            return EI_IMPULSE_CANCELED;
        }

        result->timing.dsp_us = ei_read_timer_us() - dsp_start_us;
        result->timing.dsp = (int)(result->timing.dsp_us / 1000);

        if (debug)
        {
            ei_printf("Features (%d ms.): ", result->timing.dsp);
            for (size_t ix = 0; ix < features_matrix.cols; ix++)
            {
                ei_printf_float((features_matrix.buffer[ix] - EI_CLASSIFIER_TFLITE_INPUT_ZEROPOINT) * EI_CLASSIFIER_TFLITE_INPUT_SCALE);
                ei_printf(" ");
            }
            ei_printf("\n");
        }

        ctx_start_us = ei_read_timer_us();

#if (EI_CLASSIFIER_COMPILED == 1)
        EI_IMPULSE_ERROR run_res = inference_tflite_run(ctx_start_us, output,
#if EI_CLASSIFIER_OBJDET_HAS_SCORE_TENSOR
                                                        output_labels,
                                                        output_scores,
#endif
                                                        static_cast<uint8_t *>(p_tensor_arena.get()),
                                                        result, debug);

        if (run_res != EI_IMPULSE_OK)
        {
            return run_res;
        }

        result->timing.classification_us = ei_read_timer_us() - ctx_start_us;

        return EI_IMPULSE_OK;
#endif // EI_CLASSIFIER_INFERENCING_ENGINE != EI_CLASSIFIER_TFLITE
    }

#endif // #if EI_CLASSIFIER_TFLITE_INPUT_QUANTIZED == 1 && (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TFLITE || EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TENSAIFLOW)

#if EIDSP_SIGNAL_C_FN_POINTER == 0

    /**
     * Run the impulse, if you provide an instance of sampler it will also persist the data for you
     * @param sampler Instance to an **initialized** sampler
     * @param result Object to store the results in
     * @param data_fn Function to retrieve data from sensors
     * @param debug Whether to log debug messages (default false)
     */
    __attribute__((unused)) EI_IMPULSE_ERROR run_impulse(
#if defined(EI_CLASSIFIER_HAS_SAMPLER) && EI_CLASSIFIER_HAS_SAMPLER == 1
        EdgeSampler *sampler,
#endif
        ei_impulse_result_t *result,
#ifdef __MBED__
        mbed::Callback<void(float *, size_t)> data_fn,
#else
        std::function<void(float *, size_t)> data_fn,
#endif
        bool debug = false)
    {

        float *x = (float *)calloc(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(float));
        if (!x)
        {
            return EI_IMPULSE_OUT_OF_MEMORY;
        }

        uint64_t next_tick = 0;

        uint64_t sampling_us_start = ei_read_timer_us();

        // grab some data
        for (int i = 0; i < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; i += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME)
        {
            uint64_t curr_us = ei_read_timer_us() - sampling_us_start;

            next_tick = curr_us + (EI_CLASSIFIER_INTERVAL_MS * 1000);

            data_fn(x + i, EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);

            if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED)
            {
                free(x);
                return EI_IMPULSE_CANCELED;
            }

            while (next_tick > ei_read_timer_us() - sampling_us_start)
                ;
        }

        result->timing.sampling = (ei_read_timer_us() - sampling_us_start) / 1000;

        signal_t signal;
        int err = numpy::signal_from_buffer(x, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
        if (err != 0)
        {
            free(x);
            ei_printf("ERR: signal_from_buffer failed (%d)\n", err);
            return EI_IMPULSE_DSP_ERROR;
        }

        EI_IMPULSE_ERROR r = run_classifier(&signal, result, debug);
        free(x);
        return r;
    }

#endif // #if EIDSP_SIGNAL_C_FN_POINTER == 0

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _EDGE_IMPULSE_RUN_CLASSIFIER_H_

// End of ei_run_classifier.h ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#endif // _INFERENCE_H

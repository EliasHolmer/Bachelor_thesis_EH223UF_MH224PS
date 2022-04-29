#include <Arduino.h>
#include <PDM.h>
#include <stdint.h>
#include <vector>

#include "speechpy.hpp"


static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = true; // Set this to true to see e.g. features generated from the raw signal

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

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

typedef struct {
    size_t n_output_features;
    int (*extract_fn)(signal_t *signal, matrix_t *output_matrix, void *config, const float frequency);
    void *config;
    uint8_t *axes;
    size_t axes_size;
} model_dsp_t;

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
}
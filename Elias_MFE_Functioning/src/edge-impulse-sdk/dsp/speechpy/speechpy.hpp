#ifndef _EIDSP_SPEECHPY_SPEECHPY_H_
#define _EIDSP_SPEECHPY_SPEECHPY_H_

#include "../config.hpp"
#include "../numpy.hpp"
#include <math.h>
#include "../returntypes.hpp"
#include <vector>
#include <stdint.h>
#include "../memory.hpp"

#endif // _EIDSP_SPEECHPY_SPEECHPY_H_





#ifndef _EIDSP_SPEECHPY_PROCESSING_H_
#define _EIDSP_SPEECHPY_PROCESSING_H_

































namespace ei {
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
            _prev_buffer = (float*)ei_dsp_calloc(shift * sizeof(float), 1);
            _end_of_signal_buffer = (float*)ei_dsp_calloc(shift * sizeof(float), 1);
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
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
            }
            if (offset + length > _signal->total_length) {
                EIDSP_ERR(EIDSP_OUT_OF_BOUNDS);
            }

            int ret;
            if (static_cast<int32_t>(offset) - _shift >= 0) {
                ret = _signal->get_data(offset - _shift, _shift, _prev_buffer);
                if (ret != 0) {
                    EIDSP_ERR(ret);
                }
            }
            // else we'll use the end_of_signal_buffer; so no need to check

            ret = _signal->get_data(offset, length, out_buffer);
            if (ret != 0) {
                EIDSP_ERR(ret);
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
                    EIDSP_ERR(ret);
                }
            }

            return EIDSP_OK;
        }

        ~preemphasis() {
            if (_prev_buffer) {
                ei_dsp_free(_prev_buffer, _shift * sizeof(float));
            }
            if (_end_of_signal_buffer) {
                ei_dsp_free(_end_of_signal_buffer, _shift * sizeof(float));
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
     * Preemphasising on the signal. This modifies the signal in place!
     * For memory consumption reasons you **probably** want the preemphasis class,
     * which lazily loads the signal in.
     * @param signal (array): The input signal.
     * @param shift (int): The shift step.
     * @param cof (float): The preemphasising coefficient. 0 equals to no filtering.
     * @returns 0 when successful
     */
    __attribute__((unused)) static int preemphasis(float *signal, size_t signal_size, int shift = 1, float cof = 0.98f)
    {
        if (shift < 0) {
            shift = signal_size + shift;
        }

        // so we need to keep some history
        float *prev_buffer = (float*)ei_dsp_calloc(shift * sizeof(float), 1);

        // signal - cof * xt::roll(signal, shift)
        for (size_t ix = 0; ix < signal_size; ix++) {
            float now = signal[ix];

            // under shift? read from end
            if (ix < static_cast<uint32_t>(shift)) {
                signal[ix] = now - (cof * signal[signal_size - shift + ix]);
            }
            // otherwise read from history buffer
            else {
                signal[ix] = now - (cof * prev_buffer[0]);
            }

            // roll through and overwrite last element
            numpy::roll(prev_buffer, shift, -1);
            prev_buffer[shift - 1] = now;
        }

        ei_dsp_free(prev_buffer, shift * sizeof(float));

        return EIDSP_OK;
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
     * Calculate the length of a signal that will be sused for the settings provided.
     * @param signal_size: The number of frames in the signal
     * @param sampling_frequency (int): The sampling frequency of the signal.
     * @param frame_length (float): The length of the frame in second.
     * @param frame_stride (float): The stride between frames.
     * @returns Number of frames required, or a negative number if an error occured
     */
    static int calculate_signal_used(
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
        volatile int len_sig;

        if (zero_padding) {
            // Calculation of number of frames
            numframes = static_cast<int>(
                ceil(static_cast<float>(signal_size - length) / frame_stride));

            // Zero padding
            len_sig = static_cast<int>(static_cast<float>(numframes) * frame_stride) + frame_sample_length;
        }
        else {
            numframes = static_cast<int>(
                floor(static_cast<float>(signal_size - length) / frame_stride));
            len_sig = static_cast<int>(
                (static_cast<float>(numframes - 1) * frame_stride + frame_sample_length));
        }

        return len_sig;
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
            EIDSP_ERR(EIDSP_SIGNAL_SIZE_MISMATCH);
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

        return EIDSP_OK;
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
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        int r = numpy::rfft(frame, frame_size, out_buffer, out_buffer_size, fft_points);
        if (r != EIDSP_OK) {
            return r;
        }

        for (size_t ix = 0; ix < out_buffer_size; ix++) {
            out_buffer[ix] = (1.0 / static_cast<float>(fft_points)) *
                (out_buffer[ix] * out_buffer[ix]);
        }

        return EIDSP_OK;
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
            return EIDSP_OK;
        }

        uint16_t pad_size = (win_size - 1) / 2;

        int ret;
        float *features_buffer_ptr;

        // mean & variance normalization
        EI_DSP_MATRIX(vec_pad, features_matrix->rows + (pad_size * 2), features_matrix->cols);
        if (!vec_pad.buffer) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        ret = numpy::pad_1d_symmetric(features_matrix, &vec_pad, pad_size, pad_size);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        EI_DSP_MATRIX(mean_matrix, vec_pad.cols, 1);
        if (!mean_matrix.buffer) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        EI_DSP_MATRIX(window_variance, vec_pad.cols, 1);
        if (!window_variance.buffer) {
            return EIDSP_OUT_OF_MEM;
        }

        for (size_t ix = 0; ix < features_matrix->rows; ix++) {
            // create a slice on the vec_pad
            EI_DSP_MATRIX_B(window, win_size, vec_pad.cols, vec_pad.buffer + (ix * vec_pad.cols));
            if (!window.buffer) {
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
            }

            ret = numpy::mean_axis0(&window, &mean_matrix);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(ret);
            }

            // subtract the mean for the features
            for (size_t fm_col = 0; fm_col < features_matrix->cols; fm_col++) {
                features_matrix->buffer[(ix * features_matrix->cols) + fm_col] =
                    features_matrix->buffer[(ix * features_matrix->cols) + fm_col] - mean_matrix.buffer[fm_col];
            }
        }

        ret = numpy::pad_1d_symmetric(features_matrix, &vec_pad, pad_size, pad_size);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        for (size_t ix = 0; ix < features_matrix->rows; ix++) {
            // create a slice on the vec_pad
            EI_DSP_MATRIX_B(window, win_size, vec_pad.cols, vec_pad.buffer + (ix * vec_pad.cols));
            if (!window.buffer) {
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
            }

            if (variance_normalization == true) {
                ret = numpy::std_axis0(&window, &window_variance);
                if (ret != EIDSP_OK) {
                    EIDSP_ERR(ret);
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
            if (ret != EIDSP_OK) {
                EIDSP_ERR(ret);
            }
        }

        return EIDSP_OK;
    }

    /**
     * Perform normalization for MFE frames, this converts the signal to dB,
     * then add a hard filter, and quantize / dequantize the output
     * @param features_matrix input feature matrix, will be modified in place
     */
    static int mfe_normalization(matrix_t *features_matrix, int noise_floor_db) {
        const float noise = static_cast<float>(noise_floor_db * -1);
        const float noise_scale = 1.0f / (static_cast<float>(noise_floor_db * -1) + 12.0f);

        for (size_t ix = 0; ix < features_matrix->rows * features_matrix->cols; ix++) {
            float f = features_matrix->buffer[ix];
            if (f < 1e-30) {
                f = 1e-30;
            }
            f = numpy::log10(f);
            f *= 10.0f; // scale by 10
            f += noise;
            f *= noise_scale;
            // clip again
            if (f < 0.0f) f = 0.0f;
            else if (f > 1.0f) f = 1.0f;
            features_matrix->buffer[ix] = f;
        }

        return EIDSP_OK;
    }

    /**
     * Perform normalization for spectrogram frames, this converts the signal to dB,
     * then add a hard filter
     * @param features_matrix input feature matrix, will be modified in place
     */
    static int spectrogram_normalization(matrix_t *features_matrix, int noise_floor_db) {
        const float noise = static_cast<float>(noise_floor_db * -1);
        const float noise_scale = 1.0f / (static_cast<float>(noise_floor_db * -1) + 12.0f);

        for (size_t ix = 0; ix < features_matrix->rows * features_matrix->cols; ix++) {
            float f = features_matrix->buffer[ix];
            if (f < 1e-30) {
                f = 1e-30;
            }
            f = numpy::log10(f);
            f *= 10.0f; // scale by 10
            f += noise;
            f *= noise_scale;
            // clip again
            if (f < 0.0f) f = 0.0f;
            else if (f > 1.0f) f = 1.0f;
            features_matrix->buffer[ix] = f;
        }

        return EIDSP_OK;
    }
};

} // namespace speechpy
} // namespace ei

#endif // _EIDSP_SPEECHPY_PROCESSING_H_

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EIDSP_SPEECHPY_FUNCTIONS_H_
#define _EIDSP_SPEECHPY_FUNCTIONS_H_



namespace ei {
namespace speechpy {

class functions {
public:
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

        return EIDSP_OK;
    }
};

} // namespace speechpy
} // namespace ei

#endif // _EIDSP_SPEECHPY_FUNCTIONS_H_

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ifndef _EIDSP_SPEECHPY_FEATURE_H_
#define _EIDSP_SPEECHPY_FEATURE_H_



namespace ei {
namespace speechpy {

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
#if EIDSP_QUANTIZE_FILTERBANK
        quantized_matrix_t *filterbanks,
#else
        matrix_t *filterbanks,
#endif
        uint16_t num_filter, int coefficients, uint32_t sampling_freq,
        uint32_t low_freq, uint32_t high_freq,
        bool output_transposed = false
        )
    {
        const size_t mels_mem_size = (num_filter + 2) * sizeof(float);
        const size_t hertz_mem_size = (num_filter + 2) * sizeof(float);
        const size_t freq_index_mem_size = (num_filter + 2) * sizeof(int);

        float *mels = (float*)ei_dsp_malloc(mels_mem_size);
        if (!mels) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        if (filterbanks->rows != num_filter || filterbanks->cols != static_cast<uint32_t>(coefficients)) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }
        memset(filterbanks->buffer, 0, filterbanks->rows * filterbanks->cols * sizeof(uint8_t));


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
        float *hertz = (float*)ei_dsp_malloc(hertz_mem_size);
        if (!hertz) {
            ei_dsp_free(mels, mels_mem_size);
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
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
        ei_dsp_free(mels, mels_mem_size);

        // The frequency resolution required to put filters at the
        // exact points calculated above should be extracted.
        //  So we should round those frequencies to the closest FFT bin.
        int *freq_index = (int*)ei_dsp_malloc(freq_index_mem_size);
        if (!freq_index) {
            ei_dsp_free(hertz, hertz_mem_size);
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }
        for (uint16_t ix = 0; ix < num_filter + 2; ix++) {
            freq_index[ix] = static_cast<int>(floor((coefficients + 1) * hertz[ix] / sampling_freq));
        }
        ei_dsp_free(hertz, hertz_mem_size);

        for (size_t i = 0; i < num_filter; i++) {
            int left = freq_index[i];
            int middle = freq_index[i + 1];
            int right = freq_index[i + 2];

            EI_DSP_MATRIX(z, 1, (right - left + 1));
            if (!z.buffer) {
                ei_dsp_free(freq_index, freq_index_mem_size);
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
            }
            numpy::linspace(left, right, (right - left + 1), z.buffer);
            functions::triangle(z.buffer, (right - left + 1), left, middle, right);

            // so... z now contains some values that we need to overwrite in the filterbank
            for (int zx = 0; zx < (right - left + 1); zx++) {
                size_t index = (i * filterbanks->cols) + (left + zx);

                if (output_transposed) {
                    index = ((left + zx) * filterbanks->rows) + i;
                }

                filterbanks->buffer[index] = numpy::quantize_zero_one(z.buffer[zx]);

            }
        }

        if (output_transposed) {
            uint16_t r = filterbanks->rows;
            filterbanks->rows = filterbanks->cols;
            filterbanks->cols = r;
        }

        ei_dsp_free(freq_index, freq_index_mem_size);

        return EIDSP_OK;
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
            EIDSP_ERR(ret);
        }

        if (stack_frame_info.frame_ixs->size() != out_features->rows) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (num_filters != out_features->cols) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (stack_frame_info.frame_ixs->size() != out_energies->rows || out_energies->cols != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        for (uint32_t i = 0; i < out_features->rows * out_features->cols; i++) {
            *(out_features->buffer + i) = 0;
        }

        uint16_t coefficients = fft_length / 2 + 1;

        // calculate the filterbanks first... preferably I would want to do the matrix multiplications
        // whenever they happen, but OK...
        EI_DSP_QUANTIZED_MATRIX(filterbanks, num_filters, coefficients, &numpy::dequantize_zero_one);

        if (!filterbanks.buffer) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        ret = feature::filterbanks(
            &filterbanks, num_filters, coefficients, sampling_frequency, low_frequency, high_frequency, true);
        if (ret != 0) {
            EIDSP_ERR(ret);
        }
        for (size_t ix = 0; ix < stack_frame_info.frame_ixs->size(); ix++) {
            size_t power_spectrum_frame_size = (fft_length / 2 + 1);

            EI_DSP_MATRIX(power_spectrum_frame, 1, power_spectrum_frame_size);
            if (!power_spectrum_frame.buffer) {
                EIDSP_ERR(EIDSP_OUT_OF_MEM);
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
                EIDSP_ERR(ret);
            }

            ret = processing::power_spectrum(
                signal_frame.buffer,
                stack_frame_info.frame_length,
                power_spectrum_frame.buffer,
                power_spectrum_frame_size,
                fft_length
            );

            if (ret != 0) {
                EIDSP_ERR(ret);
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
                EIDSP_ERR(ret);
            }
        }

        functions::zero_handling(out_features);

        return EIDSP_OK;
    }

    /**
     * Compute spectrogram from a sensor signal.
     * @param out_features Use `calculate_mfe_buffer_size` to allocate the right matrix.
     * @param signal: audio signal structure with functions to retrieve data from a signal
     * @param sampling_frequency (int): the sampling frequency of the signal
     *     we are working with.
     * @param frame_length (float): the length of each frame in seconds.
     *     Default is 0.020s
     * @param frame_stride (float): the step between successive frames in seconds.
     *     Default is 0.02s (means no overlap)
     * @param fft_length (int): number of FFT points. Default is 512.
     * @EIDSP_OK if OK
     */
    static int spectrogram(matrix_t *out_features,
        signal_t *signal, float sampling_frequency,
        float frame_length, float frame_stride, uint16_t fft_length,
        uint16_t version
        )
    {
        int ret = 0;

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
            EIDSP_ERR(ret);
        }

        if (stack_frame_info.frame_ixs->size() != out_features->rows) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        uint16_t coefficients = fft_length / 2 + 1;

        if (coefficients != out_features->cols) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        for (uint32_t i = 0; i < out_features->rows * out_features->cols; i++) {
            *(out_features->buffer + i) = 0;
        }

        for (size_t ix = 0; ix < stack_frame_info.frame_ixs->size(); ix++) {
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
                EIDSP_ERR(ret);
            }

            // normalize data (only when version is above 3)
            if (version >= 3) {
                // it might be that everything is already normalized here...
                bool all_between_min_1_and_1 = true;
                for (size_t ix = 0; ix < signal_frame.rows * signal_frame.cols; ix++) {
                    if (signal_frame.buffer[ix] < -1.0f || signal_frame.buffer[ix] > 1.0f) {
                        all_between_min_1_and_1 = false;
                        break;
                    }
                }

                if (!all_between_min_1_and_1) {
                    ret = numpy::scale(&signal_frame, 1.0f / 32768.0f);
                    if (ret != 0) {
                        EIDSP_ERR(ret);
                    }
                }
            }

            ret = processing::power_spectrum(
                signal_frame.buffer,
                stack_frame_info.frame_length,
                out_features->buffer + (ix * coefficients),
                coefficients,
                fft_length
            );

            if (ret != 0) {
                EIDSP_ERR(ret);
            }
        }

        functions::zero_handling(out_features);

        return EIDSP_OK;
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
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
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
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        int ret = EIDSP_OK;

        // allocate some memory for the MFE result
        EI_DSP_MATRIX(features_matrix, mfe_matrix_size.rows, mfe_matrix_size.cols);
        if (!features_matrix.buffer) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        EI_DSP_MATRIX(energy_matrix, mfe_matrix_size.rows, 1);
        if (!energy_matrix.buffer) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        ret = mfe(&features_matrix, &energy_matrix, signal,
            sampling_frequency, frame_length, frame_stride, num_filters, fft_length,
            low_frequency, high_frequency, version);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        // ok... now we need to calculate the MFCC from this...
        // first do log() over all features...
        ret = numpy::log(&features_matrix);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        // now do DST type 2
        ret = numpy::dct2(&features_matrix, DCT_NORMALIZATION_ORTHO);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
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

        return EIDSP_OK;
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
};

} // namespace speechpy
} // namespace ei

#endif // _EIDSP_SPEECHPY_FEATURE_H_

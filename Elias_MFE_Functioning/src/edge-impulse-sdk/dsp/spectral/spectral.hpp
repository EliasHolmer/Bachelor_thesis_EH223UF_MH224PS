
#ifndef _EIDSP_SPECTRAL_SPECTRAL_H_
#define _EIDSP_SPECTRAL_SPECTRAL_H_

#include "../config.hpp"
#include <vector>
#include <algorithm>
#include "../numpy.hpp"
#include <stdint.h>
#include <cmath>
#include <limits>
#include <math.h>


#endif // _EIDSP_SPECTRAL_SPECTRAL_H_

#ifndef _EIDSP_SPECTRAL_FILTERS_H_
#define _EIDSP_SPECTRAL_FILTERS_H_



#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif // M_PI

namespace ei {
namespace spectral {
namespace filters {
    /**
     * The Butterworth filter has maximally flat frequency response in the passband.
     * @param filter_order Even filter order (between 2..8)
     * @param sampling_freq Sample frequency of the signal
     * @param cutoff_freq Cut-off frequency of the signal
     * @param src Source array
     * @param dest Destination array
     * @param size Size of both source and destination arrays
     */
    static void butterworth_lowpass(
        int filter_order,
        float sampling_freq,
        float cutoff_freq,
        const float *src,
        float *dest,
        size_t size)
    {
        int n_steps = filter_order / 2;
        float a = tan(M_PI * cutoff_freq / sampling_freq);
        float a2 = pow(a, 2);
        float *A = (float*)ei_calloc(n_steps, sizeof(float));
        float *d1 = (float*)ei_calloc(n_steps, sizeof(float));
        float *d2 = (float*)ei_calloc(n_steps, sizeof(float));
        float *w0 = (float*)ei_calloc(n_steps, sizeof(float));
        float *w1 = (float*)ei_calloc(n_steps, sizeof(float));
        float *w2 = (float*)ei_calloc(n_steps, sizeof(float));

        // Calculate the filter parameters
        for(int ix = 0; ix < n_steps; ix++) {
            float r = sin(M_PI * ((2.0 * ix) + 1.0) / (2.0 * filter_order));
            sampling_freq = a2 + (2.0 * a * r) + 1.0;
            A[ix] = a2 / sampling_freq;
            d1[ix] = 2.0 * (1 - a2) / sampling_freq;
            d2[ix] = -(a2 - (2.0 * a * r) + 1.0) / sampling_freq;
        }

        // Apply the filter
        for (size_t sx = 0; sx < size; sx++) {
            dest[sx] = src[sx];

            for (int i = 0; i < n_steps; i++) {
                w0[i] = d1[i] * w1[i] + d2[i] * w2[i] + dest[sx];
                dest[sx] = A[i] * (w0[i] + (2.0 * w1[i]) + w2[i]);
                w2[i] = w1[i];
                w1[i] = w0[i];
            }
        }

        ei_free(A);
        ei_free(d1);
        ei_free(d2);
        ei_free(w0);
        ei_free(w1);
        ei_free(w2);
    }

    /**
     * The Butterworth filter has maximally flat frequency response in the passband.
     * @param filter_order Even filter order (between 2..8)
     * @param sampling_freq Sample frequency of the signal
     * @param cutoff_freq Cut-off frequency of the signal
     * @param src Source array
     * @param dest Destination array
     * @param size Size of both source and destination arrays
     */
    static void butterworth_highpass(
        int filter_order,
        float sampling_freq,
        float cutoff_freq,
        const float *src,
        float *dest,
        size_t size)
    {
        int n_steps = filter_order / 2;
        float a = tan(M_PI * cutoff_freq / sampling_freq);
        float a2 = pow(a, 2);
        float *A = (float*)ei_calloc(n_steps, sizeof(float));
        float *d1 = (float*)ei_calloc(n_steps, sizeof(float));
        float *d2 = (float*)ei_calloc(n_steps, sizeof(float));
        float *w0 = (float*)ei_calloc(n_steps, sizeof(float));
        float *w1 = (float*)ei_calloc(n_steps, sizeof(float));
        float *w2 = (float*)ei_calloc(n_steps, sizeof(float));

        // Calculate the filter parameters
        for (int ix = 0; ix < n_steps; ix++) {
            float r = sin(M_PI * ((2.0 * ix) + 1.0) / (2.0 * filter_order));
            sampling_freq = a2 + (2.0 * a * r) + 1.0;
            A[ix] = 1.0f / sampling_freq;
            d1[ix] = 2.0 * (1 - a2) / sampling_freq;
            d2[ix] = -(a2 - (2.0 * a * r) + 1.0) / sampling_freq;
        }

        // Apply the filter
        for (size_t sx = 0; sx < size; sx++) {
            dest[sx] = src[sx];

            for (int i = 0; i < n_steps; i++) {
                w0[i] = d1[i] * w1[i] + d2[i] * w2[i] + dest[sx];
                dest[sx] = A[i] * (w0[i] - (2.0 * w1[i]) + w2[i]);
                w2[i] = w1[i];
                w1[i] = w0[i];
            }
        }

        ei_free(A);
        ei_free(d1);
        ei_free(d2);
        ei_free(w0);
        ei_free(w1);
        ei_free(w2);
    }

} // namespace filters
} // namespace spectral
} // namespace ei

#endif // _EIDSP_SPECTRAL_FILTERS_H_


#ifndef _EIDSP_SPECTRAL_PROCESSING_H_
#define _EIDSP_SPECTRAL_PROCESSING_H_

namespace ei {
namespace spectral {

namespace processing {
    /**
     * Scaling on the signal.
     * @param signal: The input signal.
     * @param scaling (int): To scale by which factor (e.g. 10 here means multiply by 10)
     */
    class scale {
public:
        scale(ei_signal_t *signal, float scaling = 1.0f)
            : _signal(signal), _scaling(scaling)
        {
        }

        /**
         * Get scaled data from the underlying sensor buffer...
         * This retrieves data from the signal then scales it.
         * @param offset Offset in the audio signal
         * @param length Length of the audio signal
         */
        int get_data(size_t offset, size_t length, float *out_buffer) {
            if (offset + length > _signal->total_length) {
                EIDSP_ERR(EIDSP_OUT_OF_BOUNDS);
            }

            int ret = _signal->get_data(offset, length, out_buffer);
            if (ret != 0) {
                EIDSP_ERR(ret);
            }

            EI_DSP_MATRIX_B(temp, 1, length, out_buffer);
            return numpy::scale(&temp, _scaling);
        }

private:
        ei_signal_t *_signal;
        float _scaling;
    };
}

namespace processing {
    typedef struct {
        float freq;
        float amplitude;
    } freq_peak_t;

    typedef struct {
        EIDSP_i16 freq;
        EIDSP_i16 amplitude;
    } freq_peak_i16_t;

    typedef struct {
        EIDSP_i32 freq;
        EIDSP_i32 amplitude;
    } freq_peak_i32_t;

    /**
     * Scale a the signal. This modifies the signal in place!
     * For memory consumption reasons you **probably** want the scaling class,
     * which lazily loads the signal in.
     * @param signal (array): The input signal.
     * @param signal_size: The length of the signal.
     * @param scale (float): The scaling factor (multiplies by this number).
     * @returns 0 when successful
     */
    __attribute__((unused)) static int scale(float *signal, size_t signal_size, float scale = 1)
    {
        EI_DSP_MATRIX_B(temp, 1, signal_size, signal);
        return numpy::scale(&temp, scale);
    }

    /**
     * Filter data along one-dimension with an IIR or FIR filter using
     * Butterworth digital and analog filter design.
     * This modifies the matrix in-place (per row)
     * @param matrix Input matrix
     * @param sampling_freq Sampling frequency
     * @param filter_cutoff
     * @param filter_order
     * @returns 0 when successful
     */
    static int butterworth_lowpass_filter(
        matrix_t *matrix,
        float sampling_frequency,
        float filter_cutoff,
        uint8_t filter_order)
    {
        for (size_t row = 0; row < matrix->rows; row++) {
            filters::butterworth_lowpass(
                filter_order,
                sampling_frequency,
                filter_cutoff,
                matrix->buffer + (row * matrix->cols),
                matrix->buffer + (row * matrix->cols),
                matrix->cols);
        }

        return EIDSP_OK;
    }

    /**
     * Filter data along one-dimension with an IIR or FIR filter using
     * Butterworth digital and analog filter design.
     * This modifies the matrix in-place (per row)
     * @param matrix Input matrix
     * @param sampling_freq Sampling frequency
     * @param filter_cutoff
     * @param filter_order
     * @returns 0 when successful
     */
    static int butterworth_highpass_filter(
        matrix_t *matrix,
        float sampling_frequency,
        float filter_cutoff,
        uint8_t filter_order)
    {
        for (size_t row = 0; row < matrix->rows; row++) {
            filters::butterworth_highpass(
                filter_order,
                sampling_frequency,
                filter_cutoff,
                matrix->buffer + (row * matrix->cols),
                matrix->buffer + (row * matrix->cols),
                matrix->cols);
        }

        return EIDSP_OK;
    }

    /**
     * Find peaks in a FFT spectrum
     * threshold is *normalized* threshold
     * (I'm still not completely sure if this matches my Python code but it looks OK)
     * @param input_matrix Matrix with FFT data of size 1xM
     * @param output_matrix Output matrix with N rows for every peak you want to find.
     * @param threshold Minimum threshold
     * @param peaks_found Out parameter with the number of peaks found
     * @returns 0 if OK
     */
    static int find_peak_indexes(
        matrix_t *input_matrix,
        matrix_t *output_matrix,
        float threshold,
        uint16_t *peaks_found)
    {
        if (input_matrix->rows != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (output_matrix->cols != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        uint16_t out_ix = 0;
        size_t in_size = input_matrix->cols;
        float *in = input_matrix->buffer;
        size_t out_size = output_matrix->rows;
        float *out = output_matrix->buffer;

        // for normalized threshold calculation
        float min = FLT_MAX, max = 0.0f;
        for (size_t ix = 0; ix < in_size - 1; ix++) {
            if (in[ix] < min) {
                min = in[ix];
            }
            if (in[ix] > max) {
                max = in[ix];
            }
        }


        float prev = in[0];

        // so....
        for (size_t ix = 1; ix < in_size - 1; ix++) {
            // first make sure it's actually a peak...
            if (in[ix] > prev && in[ix] > in[ix+1]) {
                // then make sure the threshold is met (on both?)
                float height = (in[ix] - prev) + (in[ix] - in[ix + 1]);
                // printf("%d inx: %f height: %f threshold: %f\r\n", ix, in[ix], height, threshold);
                if (height > threshold) {
                    out[out_ix] = ix;
                    out_ix++;
                    if (out_ix == out_size) break;
                }
            }

            prev = in[ix];
        }

        *peaks_found = out_ix;

        return EIDSP_OK;
    }

    /**
     * Find peaks in FFT
     * @param fft_matrix Matrix of FFT numbers (1xN)
     * @param output_matrix Matrix for the output (Mx2), one row per output you want and two colums per row
     * @param sampling_freq How often we sample (in Hz)
     * @param threshold Minimum threshold (default: 0.1)
     * @returns
     */
    static int find_fft_peaks(
        matrix_t *fft_matrix,
        matrix_t *output_matrix,
        float sampling_freq,
        float threshold,
        uint16_t fft_length)
    {
        if (fft_matrix->rows != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (output_matrix->cols != 2) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (output_matrix->rows == 0) {
            return EIDSP_OK;
        }

        int ret;

        int N = static_cast<int>(fft_length);
        float T = 1.0f / sampling_freq;

        EI_DSP_MATRIX(freq_space, 1, fft_matrix->cols);
        ret = numpy::linspace(0.0f, 1.0f / (2.0f * T), floor(N / 2), freq_space.buffer);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        EI_DSP_MATRIX(peaks_matrix, output_matrix->rows * 10, 1);

        uint16_t peak_count;
        ret = find_peak_indexes(fft_matrix, &peaks_matrix, 0.0f, &peak_count);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        // turn this into C++ vector and sort it based on amplitude
        std::vector<freq_peak_t> peaks;
        for (uint8_t ix = 0; ix < peak_count; ix++) {
            freq_peak_t d;

            d.freq = freq_space.buffer[static_cast<uint32_t>(peaks_matrix.buffer[ix])];
            d.amplitude = fft_matrix->buffer[static_cast<uint32_t>(peaks_matrix.buffer[ix])];
            // printf("freq %f : %f amp: %f\r\n", peaks_matrix.buffer[ix], d.freq, d.amplitude);
            if (d.amplitude < threshold) {
                d.freq = 0.0f;
                d.amplitude = 0.0f;
            }
            peaks.push_back(d);
        }
        sort(peaks.begin(), peaks.end(),
            [](const freq_peak_t & a, const freq_peak_t & b) -> bool
        {
            return a.amplitude > b.amplitude;
        });

        // fill with zeros at the end (if needed)
        for (size_t ix = peaks.size(); ix < output_matrix->rows; ix++) {
            freq_peak_t d;
            d.freq = 0;
            d.amplitude = 0;
            peaks.push_back(d);
        }

        for (size_t row = 0; row < output_matrix->rows; row++) {
            // col 0 is freq, col 1 is ampl
            output_matrix->buffer[row * output_matrix->cols + 0] = peaks[row].freq;
            output_matrix->buffer[row * output_matrix->cols + 1] = peaks[row].amplitude;
        }

        return EIDSP_OK;
    }


    /**
     * Calculate spectral power edges in a singal
     * @param fft_matrix FFT matrix (1xM)
     * @param input_matrix_cols Number of columns in the input matrix
     * @param edges_matrix The power edges (Nx1) where N=is number of edges
     *      (e.g. [0.1, 0.5, 1.0, 2.0, 5.0])
     * @param output_matrix Output matrix of size (N-1 x 1)
     * @param sampling_freq Sampling frequency
     * @returns 0 if OK
     */
    int spectral_power_edges(
        matrix_t *fft_matrix,
        matrix_t *freq_matrix,
        matrix_t *edges_matrix,
        matrix_t *output_matrix,
        float sampling_freq
    ) {
        if (fft_matrix->rows != 1 || freq_matrix->rows != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (edges_matrix->cols != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (output_matrix->rows != edges_matrix->rows - 1 || output_matrix->cols != edges_matrix->cols) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (fft_matrix->cols != freq_matrix->cols) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        EI_DSP_MATRIX(buckets, 1, edges_matrix->rows - 1);
        EI_DSP_MATRIX(bucket_count, 1, edges_matrix->rows - 1);

        for (uint16_t ix = 0; ix < freq_matrix->cols; ix++) {
            float t = freq_matrix->buffer[ix];
            float v = fft_matrix->buffer[ix];

            // does this fit between any edges?
            for (uint16_t ex = 0; ex < edges_matrix->rows - 1; ex++) {
                if (t >= edges_matrix->buffer[ex] && t < edges_matrix->buffer[ex + 1]) {
                    buckets.buffer[ex] += v;
                    bucket_count.buffer[ex]++;
                    break;
                }
            }
        }

        // average out and push to vector
        for (uint16_t ex = 0; ex < edges_matrix->rows - 1; ex++) {
            if (bucket_count.buffer[ex] == 0.0f) {
                output_matrix->buffer[ex] = 0.0f;
            }
            else {
                output_matrix->buffer[ex] = buckets.buffer[ex] / bucket_count.buffer[ex];
            }
        }

        return EIDSP_OK;
    }


    /**
     * Estimate power spectral density using a periodogram using Welch's method.
     * @param input_matrix Of size 1xN
     * @param out_fft_matrix Output matrix of size 1x(n_fft/2+1) with frequency data
     * @param out_freq_matrix Output matrix of size 1x(n_fft/2+1) with frequency data
     * @param sampling_freq The sampling frequency
     * @param n_fft Number of FFT buckets
     * @returns 0 if OK
     */
    int periodogram(matrix_t *input_matrix, matrix_t *out_fft_matrix, matrix_t *out_freq_matrix, float sampling_freq, uint16_t n_fft)
    {
        if (input_matrix->rows != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (out_fft_matrix->rows != 1 || out_fft_matrix->cols != static_cast<uint32_t>(n_fft / 2 + 1)) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (out_freq_matrix->rows != 1 || out_freq_matrix->cols != static_cast<uint32_t>(n_fft / 2 + 1)) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (input_matrix->buffer == NULL) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        if (out_fft_matrix->buffer == NULL) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        if (out_freq_matrix->buffer == NULL) {
            EIDSP_ERR(EIDSP_OUT_OF_MEM);
        }

        // map over the input buffer, so we can manipulate the number of columns
        EI_DSP_MATRIX_B(welch_matrix, input_matrix->rows, input_matrix->cols, input_matrix->buffer);

        uint16_t nperseg = n_fft;

        if (n_fft > input_matrix->cols) {
            nperseg = input_matrix->cols;
        }
        // make the column align to nperseg in this case
        else if (n_fft < input_matrix->cols) {
            welch_matrix.cols = n_fft;
        }

        EI_DSP_MATRIX(triage_segments, 1, nperseg);
        for (uint16_t ix = 0; ix < nperseg; ix++) {
            triage_segments.buffer[ix] = 1.0f;
        }

        float scale = 1.0f / (sampling_freq * nperseg);

        for (uint16_t ix = 0; ix < n_fft / 2 + 1; ix++) {
            out_freq_matrix->buffer[ix] = static_cast<float>(ix) * (1.0f / (n_fft * (1.0f / sampling_freq)));
        }

        int ret;

        // now we need to detrend... which is done constant so just subtract the mean
        EI_DSP_MATRIX(mean_matrix, 1, 1);
        ret = numpy::mean(&welch_matrix, &mean_matrix);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        ret = numpy::subtract(&welch_matrix, &mean_matrix);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(ret);
        }

        fft_complex_t *fft_output = (fft_complex_t*)ei_dsp_calloc((n_fft / 2 + 1) * sizeof(fft_complex_t), 1);
        ret = numpy::rfft(welch_matrix.buffer, welch_matrix.cols, fft_output, n_fft / 2 + 1, n_fft);
        if (ret != EIDSP_OK) {
            ei_dsp_free(fft_output, (n_fft / 2 + 1) * sizeof(fft_complex_t));
            EIDSP_ERR(ret);
        }

        // conjugate and then multiply with itself and scale
        for (uint16_t ix = 0; ix < n_fft / 2 + 1; ix++) {
            fft_output[ix].r = (fft_output[ix].r * fft_output[ix].r) +
                (abs(fft_output[ix].i * fft_output[ix].i));
            fft_output[ix].i = 0.0f;

            fft_output[ix].r *= scale;

            if (ix != n_fft / 2) {
                fft_output[ix].r *= 2;
            }

            // then multiply by itself...
            out_fft_matrix->buffer[ix] = fft_output[ix].r;
        }

        ei_dsp_free(fft_output, (n_fft / 2 + 1) * sizeof(fft_complex_t));

        return EIDSP_OK;
    }
} // namespace processing
} // namespace spectral
} // namespace ei

#endif // _EIDSP_SPECTRAL_PROCESSING_H_

#ifndef _EIDSP_SPECTRAL_FEATURE_H_
#define _EIDSP_SPECTRAL_FEATURE_H_



namespace ei {
namespace spectral {

typedef enum {
    filter_none = 0,
    filter_lowpass = 1,
    filter_highpass = 2
} filter_t;

class feature {
public:
    /**
     * Calculate the spectral features over a signal.
     * @param out_features Output matrix. Use `calculate_spectral_buffer_size` to calculate
     *  the size required. Needs as many rows as `raw_data`.
     * @param input_matrix Signal, with one row per axis
     * @param sampling_freq Sampling frequency of the signal
     * @param filter_type Filter type
     * @param filter_cutoff Filter cutoff frequency
     * @param filter_order Filter order
     * @param fft_length Length of the FFT signal
     * @param fft_peaks Number of FFT peaks to find
     * @param fft_peaks_threshold Minimum threshold
     * @param edges_matrix Spectral power edges
     * @returns 0 if OK
     */
    static int spectral_analysis(
        matrix_t *out_features,
        matrix_t *input_matrix,
        float sampling_freq,
        filter_t filter_type,
        float filter_cutoff,
        uint8_t filter_order,
        uint16_t fft_length,
        uint8_t fft_peaks,
        float fft_peaks_threshold,
        matrix_t *edges_matrix_in
    ) {
        if (out_features->rows != input_matrix->rows) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (out_features->cols != calculate_spectral_buffer_size(true, fft_peaks, edges_matrix_in->rows)) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        if (edges_matrix_in->cols != 1) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        int ret;

        size_t axes = input_matrix->rows;

        // calculate the mean
        EI_DSP_MATRIX(mean_matrix, axes, 1);
        ret = numpy::mean(input_matrix, &mean_matrix);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        // scale by the mean
        ret = numpy::subtract(input_matrix, &mean_matrix);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        // apply filter
        if (filter_type == filter_lowpass) {
            ret = spectral::processing::butterworth_lowpass_filter(
                input_matrix, sampling_freq, filter_cutoff, filter_order);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
            }
        }
        else if (filter_type == filter_highpass) {
            ret = spectral::processing::butterworth_highpass_filter(
                input_matrix, sampling_freq, filter_cutoff, filter_order);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
            }
        }

        // calculate RMS
        EI_DSP_MATRIX(rms_matrix, axes, 1);
        ret = numpy::rms(input_matrix, &rms_matrix);
        if (ret != EIDSP_OK) {
            EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
        }

        // find peaks in FFT
        EI_DSP_MATRIX(peaks_matrix, axes, fft_peaks * 2);

        for (size_t row = 0; row < input_matrix->rows; row++) {
            // per axis code

            // get a slice of the current axis
            EI_DSP_MATRIX_B(axis_matrix, 1, input_matrix->cols, input_matrix->buffer + (row * input_matrix->cols));

            // calculate FFT
            EI_DSP_MATRIX(fft_matrix, 1, fft_length / 2 + 1);
            ret = numpy::rfft(axis_matrix.buffer, axis_matrix.cols, fft_matrix.buffer, fft_matrix.cols, fft_length);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
            }

            // multiply by 2/N
            numpy::scale(&fft_matrix, (2.0f / static_cast<float>(fft_length)));

            // we're now using the FFT matrix to calculate peaks etc.
            EI_DSP_MATRIX(peaks_matrix, fft_peaks, 2);
            ret = spectral::processing::find_fft_peaks(&fft_matrix, &peaks_matrix,
                sampling_freq, fft_peaks_threshold, fft_length);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(EIDSP_MATRIX_SIZE_MISMATCH);
            }

            // calculate periodogram for spectral power buckets
            EI_DSP_MATRIX(period_fft_matrix, 1, fft_length / 2 + 1);
            EI_DSP_MATRIX(period_freq_matrix, 1, fft_length / 2 + 1);
            ret = spectral::processing::periodogram(&axis_matrix,
                &period_fft_matrix, &period_freq_matrix, sampling_freq, fft_length);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(ret);
            }

            EI_DSP_MATRIX(edges_matrix_out, edges_matrix_in->rows - 1, 1);
            ret = spectral::processing::spectral_power_edges(
                &period_fft_matrix,
                &period_freq_matrix,
                edges_matrix_in,
                &edges_matrix_out,
                sampling_freq);
            if (ret != EIDSP_OK) {
                EIDSP_ERR(ret);
            }

            float *features_row = out_features->buffer + (row * out_features->cols);

            size_t fx = 0;

            features_row[fx++] = rms_matrix.buffer[row];
            for (size_t peak_row = 0; peak_row < peaks_matrix.rows; peak_row++) {
                features_row[fx++] = peaks_matrix.buffer[peak_row * peaks_matrix.cols + 0];
                features_row[fx++] = peaks_matrix.buffer[peak_row * peaks_matrix.cols + 1];
            }
            for (size_t edge_row = 0; edge_row < edges_matrix_out.rows; edge_row++) {
                features_row[fx++] = edges_matrix_out.buffer[edge_row * edges_matrix_out.cols] / 10.0f;
            }
        }

        return EIDSP_OK;
    }


    /**
     * Calculate the buffer size for Spectral Analysis
     * @param rms: Whether to calculate the RMS as part of the features
     * @param peaks_count: Number of FFT peaks
     * @param spectral_edges_count: Number of spectral edges
     */
    static size_t calculate_spectral_buffer_size(
        bool rms, size_t peaks_count, size_t spectral_edges_count)
    {
        size_t count = 0;
        if (rms) count++;
        count += (peaks_count * 2);
        if (spectral_edges_count > 0) {
            count += (spectral_edges_count - 1);
        }
        return count;
    }
};

} // namespace spectral
} // namespace ei



#endif // _EIDSP_SPECTRAL_FEATURE_H_

#ifndef __FIR_FILTER__H__
#define __FIR_FILTER__H__


/**
 * @brief 
 * 
 * @tparam input_t Type of input array.  Either matrix_i16_t, or matrix_i32_t
 * @tparam acc_t Accumulator size that matches above.  64bit for i16 
 */
template <class input_t, class acc_t>
class fir_filter
{
private:
    /**
     * @brief Set the taps lowpass object
     * 
     * @param cutoff_normalized Should be in the range 0..0.5 (0.5 being the nyquist)
     */
    void set_taps_lowpass(float cutoff_normalized, std::vector<float> &f_taps)
    {
        //http://www.dspguide.com/ch16/2.htm
        float sine_scale = 2 * M_PI * cutoff_normalized;
        // offset is M/2...M is filter order -1. so truncation is desired
        int offset = filter_size / 2;
        for (int i = 0; i < filter_size / 2; i++)
        {
            f_taps[i] = sin(sine_scale * (i - offset)) / (i - offset);
        }
        f_taps[filter_size / 2] = sine_scale;
        for (int i = filter_size / 2 + 1; i < filter_size; i++)
        {
            f_taps[i] = sin(sine_scale * (i - offset)) / (i - offset);
        }
    }

    void apply_hamming(std::vector<float> &f_taps)
    {
        for (int i = 0; i < filter_size; i++)
        {
            f_taps[i] *= 0.54 - 0.46 * cos(2 * M_PI * i / (filter_size - 1));
        }
    }

    void scale_to_unity_gain(std::vector<float> &f_taps)
    {
        //find the sum of taps
        float sum = 0;
        for (auto tap : f_taps)
        {
            sum += tap;
        }
        //scale down
        for (auto &tap : f_taps)
        {
            tap /= sum;
        }
    }

    void convert_lowpass_to_highpass(std::vector<float> &f_taps)
    {
        for (size_t i = 0; i < f_taps.size(); i += 2)
        {
            f_taps[i] *= -1;
        }
    }

public:
    /**
     * @brief Perform in place filtering on the input matrix
     * @param sampling_frequency Sampling freqency of data
     * @param filter_size Number of taps desired (note, filter order +1)
     * @param lowpass_cutoff Lowpass cutoff freqency.  If 0, will be a high pass filter
     * @param highpass_cutoff Highpass cutoff.  If 0, will just be a lowpass.  If both lowpass and higpass, bandpass
     * @param decimation_ratio To downsample, ratio of samples to get rid of.  
     * For example, 4 to go from sample rate of 40k to 10k.  LOWPASS CUTOFF MUST MATCH THIS
     * If you don't filter the high frequencies, they WILL alias into the passband
     * So in the above example, you would want to cutoff at 5K (so you have some buffer)
     */
    fir_filter(
        float sampling_frequency,
        uint8_t filter_size,
        float lowpass_cutoff,
        float highpass_cutoff = 0,
        int decimation_ratio = 1) :  taps(filter_size) , history(filter_size, 0)
    {
        this->filter_size = filter_size;
        std::vector<float> f_taps(filter_size, 0);
        if( highpass_cutoff == 0 && lowpass_cutoff == 0 ) 
        {
            ei_printf("You must choose either a lowpass or highpass cutoff");
            return; // return a filter that will return zeros always
        }
        if (highpass_cutoff == 0)
        {
            // use normalized frequency
            set_taps_lowpass(lowpass_cutoff / sampling_frequency, f_taps);
        }
        if (lowpass_cutoff == 0)
        {
            //for highpass, we'll just design a lowpass filter, then invert its spectrum
            set_taps_lowpass(highpass_cutoff / sampling_frequency, f_taps);
        }
        //todo bandpass
        apply_hamming(f_taps);
        //scale to unity gain in passband (this prevents overflow)
        scale_to_unity_gain(f_taps);
        // aka if highpass filter
        if (lowpass_cutoff == 0)
        {
            //now invert the spectrum
            convert_lowpass_to_highpass(f_taps);
        }
        // scale and write into fixed point taps
        for (int i = 0; i < filter_size; i++)
        {
            taps[i] = f_taps[i] * 32767;
        }
    }

/**
 * @brief Apply the filter to the input data.  You can do this blockwise, as the object preserves memory of old samples
 * Call reset if there's a gap in the data
 * 
 * @param src Source array
 * @param dest Output array (can be the same as source for in place)
 * @param size Number of samples to process
 */
    void apply_filter(
        const input_t *src,
        input_t *dest,
        size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            history[write_index] = src[i];
            int read_index = write_index;
            //minus one b/c of the sign bit
            int shift = (sizeof(input_t) * 8) - 1;
            //stuff a 1 into one less than we're going to shift to effectively round
            //this is essentially resetting the accumulator back to zero otherwise
            acc_t accumulator = 1 << (shift - 1);
            for (auto tap : taps)
            {
                accumulator += static_cast<acc_t>(tap) * history[read_index];
                //wrap the read index
                read_index = read_index == 0 ? filter_size - 1 : read_index - 1;
            }
            //wrap the write index
            write_index++;
            if (write_index == filter_size)
            {
                write_index = 0;
            }

            accumulator >>= shift;
            //saturate if overflow
            if (accumulator > std::numeric_limits<input_t>::max())
            {
                dest[i] = std::numeric_limits<input_t>::max();
            }
            else if (accumulator < std::numeric_limits<input_t>::min())
            {
                dest[i] = std::numeric_limits<input_t>::min();
            }
            else
            {
                dest[i] = accumulator;
            }
        }
    }

    /**
     * @brief Reset the filter (when changing rows for instance, for a new signal)
     * This simply clears the filter history
     * 
     */
    void reset()
    {
        std::fill(history.begin(), history.end(), 0);
    }

private:
    std::vector<input_t> taps;
    std::vector<input_t> history;
    int write_index = 0;
    int filter_size;

    friend class AccelerometerQuantizedTestCase;

};
#endif  //!__FIR_FILTER__H__

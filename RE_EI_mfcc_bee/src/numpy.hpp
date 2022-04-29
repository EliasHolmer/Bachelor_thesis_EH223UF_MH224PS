#include <Arduino.h>

#include <cfloat>
#include "constants_structs.h"
#include "kiss_fttr.h"


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
    static int dct2(float *input, size_t N, int normalization = 0) {
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
    static int dct2(matrix_t *matrix, int normalization = 0) {
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


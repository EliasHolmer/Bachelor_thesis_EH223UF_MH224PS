/* Edge Impulse inferencing library
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _EIDSP_CPP_CONFIG_H_
#define _EIDSP_CPP_CONFIG_H_

// clang-format off
#ifndef EIDSP_USE_CMSIS_DSP // __ARM_ARCH_PROFILE is a predefine of arm-gcc.  __TARGET_* is armcc
  // Mbed OS versions before 5.7 are not based on CMSIS5, disable CMSIS-DSP and CMSIS-NN instructions
        #include "mbed_version.h"
       
            #define EIDSP_USE_CMSIS_DSP      1
        // Arduino on Mbed targets prior to Mbed OS 6.0.0 ship their own CMSIS-DSP sources
            #define EIDSP_LOAD_CMSIS_DSP_SOURCES      1

#endif // ifndef EIDSP_USE_CMSIS_DSP

#define EIDSP_i32                int32_t
#define EIDSP_i16                int16_t
#define EIDSP_i8                 q7_t
#define ARM_MATH_ROUNDING        1


#ifndef EIDSP_USE_ASSERTS
#define EIDSP_USE_ASSERTS        0
#endif // EIDSP_USE_ASSERTS


#define EIDSP_ERR(err_code) return(err_code)


// To save memory you can quantize the filterbanks,
// this has an effect on runtime speed as CMSIS-DSP does not have optimized instructions
// for q7 matrix multiplication and matrix transformation...
#ifndef EIDSP_QUANTIZE_FILTERBANK
#define EIDSP_QUANTIZE_FILTERBANK    1
#endif // EIDSP_QUANTIZE_FILTERBANK

// prints buffer allocations to stdout, useful when debugging
#ifndef EIDSP_TRACK_ALLOCATIONS
#define EIDSP_TRACK_ALLOCATIONS      0
#endif // EIDSP_TRACK_ALLOCATIONS

// set EIDSP_TRACK_ALLOCATIONS=1 and EIDSP_PRINT_ALLOCATIONS=0
// to track but not print allocations
#ifndef EIDSP_PRINT_ALLOCATIONS
#define EIDSP_PRINT_ALLOCATIONS      1
#endif

#ifndef EIDSP_SIGNAL_C_FN_POINTER
#define EIDSP_SIGNAL_C_FN_POINTER    0
#endif // EIDSP_SIGNAL_C_FN_POINTER

// clang-format on
#endif // _EIDSP_CPP_CONFIG_H_

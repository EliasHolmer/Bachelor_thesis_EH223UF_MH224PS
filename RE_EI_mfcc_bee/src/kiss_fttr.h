#include "constants_structs.h"
#include "_kiss_fft_guts.h"

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
    } else {
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
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf);
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
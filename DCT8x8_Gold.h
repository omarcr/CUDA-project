/**
**************************************************************************
* \file DCT8x8_Gold.h
* \brief Contains declaration of CPU versions of DCT, IDCT and quantization
* routines.
*
* Contains declaration of CPU versions of DCT, IDCT and quantization
* routines.
*/


#pragma once

#include "BmpUtil.h"

//this functions are being implemented in the DCT8x8_Gold.cpp
//the contents of the file are included in the annex 
extern "C"
{
    void computeDCT8x8Gold1(const float *fSrc, float *fDst, int Stride, ROI Size);
    void computeIDCT8x8Gold1(const float *fSrc, float *fDst, int Stride, ROI Size);
    void quantizeGoldFloat(float *fSrcDst, int Stride, ROI Size);
    void quantizeGoldShort(short *fSrcDst, int Stride, ROI Size);
}

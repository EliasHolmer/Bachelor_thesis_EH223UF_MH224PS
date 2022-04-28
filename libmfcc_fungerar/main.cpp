#include <Arduino.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

// PI is allready defined.
//#define PI 3.141592

float GetCenterFrequency(unsigned int filterBand)
{
	float centerFrequency = 0.0f;
	float exponent;

	if(filterBand == 0)
	{
		centerFrequency = 0;
	}
	else if(filterBand >= 1 && filterBand <= 14)
	{
		centerFrequency = (200.0f * filterBand) / 3.0f;
	}
	else
	{
		exponent = filterBand - 14.0f;
		centerFrequency = pow(1.0711703, exponent);
		centerFrequency *= 1073.4;
	}
	
	return centerFrequency;
}


float GetMagnitudeFactor(unsigned int filterBand)
{
	float magnitudeFactor = 0.0f;
	
	if(filterBand >= 1 && filterBand <= 14)
	{
		magnitudeFactor = 0.015;
	}
	else if(filterBand >= 15 && filterBand <= 48)
	{
		magnitudeFactor = 2.0f / (GetCenterFrequency(filterBand + 1) - GetCenterFrequency(filterBand -1));
	}

	return magnitudeFactor;
}




float GetFilterParameter(unsigned int samplingRate, unsigned int binSize, unsigned int frequencyBand, unsigned int filterBand)
{
	float filterParameter = 0.0f;

	float boundary = (frequencyBand * samplingRate) / binSize;		// k * Fs / N
	float prevCenterFrequency = GetCenterFrequency(filterBand - 1);		// fc(l - 1) etc.
	float thisCenterFrequency = GetCenterFrequency(filterBand);
	float nextCenterFrequency = GetCenterFrequency(filterBand + 1);

	if(boundary >= 0 && boundary < prevCenterFrequency)
	{
		filterParameter = 0.0f;
	}
	else if(boundary >= prevCenterFrequency && boundary < thisCenterFrequency)
	{
		filterParameter = (boundary - prevCenterFrequency) / (thisCenterFrequency - prevCenterFrequency);
		filterParameter *= GetMagnitudeFactor(filterBand);
	}
	else if(boundary >= thisCenterFrequency && boundary < nextCenterFrequency)
	{
		filterParameter = (boundary - nextCenterFrequency) / (thisCenterFrequency - nextCenterFrequency);
		filterParameter *= GetMagnitudeFactor(filterBand);
	}
	else if(boundary >= nextCenterFrequency && boundary < samplingRate)
	{
		filterParameter = 0.0f;
	}

	return filterParameter;
}


float NormalizationFactor(int NumFilters, int m)
{
	float normalizationFactor = 0.0f;

	if(m == 0)
	{
		normalizationFactor = sqrt(1.0f / NumFilters);
	}
	else 
	{
		normalizationFactor = sqrt(2.0f / NumFilters);
	}
	
	return normalizationFactor;
}



float GetCoefficient(float* spectralData, unsigned int samplingRate, unsigned int NumFilters, unsigned int binSize, unsigned int m)
{
	float result = 0.0f;
	float outerSum = 0.0f;
	float innerSum = 0.0f;
	unsigned int k, l;

	// 0 <= m < L
	if(m >= NumFilters)
	{
		// This represents an error condition - the specified coefficient is greater than or equal to the number of filters. The behavior in this case is undefined.
		return 0.0f;
	}

	result = NormalizationFactor(NumFilters, m);

	
	for(l = 1; l <= NumFilters; l++)
	{
		// Compute inner sum
		innerSum = 0.0f;
		for(k = 0; k < binSize - 1; k++)
		{
			innerSum += fabs(spectralData[k] * GetFilterParameter(samplingRate, binSize, k, l));
		}

		if(innerSum > 0.0f)
		{
			innerSum = log(innerSum); // The log of 0 is undefined, so don't use it
		}

		innerSum = innerSum * cos(((m * PI) / NumFilters) * (l - 0.5f));

		outerSum += innerSum;
	}

	result *= outerSum;

	return result;
}



void setup() {
  // put your setup code here, to run once:

  

}

void printfloat( float val, unsigned int precision){
// prints val with number of decimal places determine by precision
// NOTE: precision is 1 followed by the number of zeros for the desired number of decimial places
// example: printfloat( 3.1415, 100); // prints 3.14 (two decimal places)

    Serial.print (int(val));  //prints the int part
    Serial.print("."); // print the decimal point
    unsigned int frac;
    if(val >= 0)
        frac = (val - int(val)) * precision;
    else
        frac = (int(val)- val ) * precision;
    Serial.println(frac,DEC) ;
} 

void loop() {
  // put your main code here, to run repeatedly:
	// Read in sample data from sample.dat
	// sample.dat contains an 8192-point spectrum from a sine wave at 440Hz (A) in float precision
	// Spectrum was computed using FFTW (http://www.fftw.org/)
	// Data was not windowed (rectangular)



	// Holds the spectrum data to be analyzed
	float spectrum[1000];
	// Pointer to the sample data file
	FILE *sampleFile;

	// Index counter - used to keep track of which data point is being read in
	int i = 0;

	// Determine which MFCC coefficient to compute
	unsigned int coeff;

  

	// Holds the value of the computed coefficient
	float mfcc_result;

	// Initialize the spectrum
	memset(&spectrum, 0, sizeof(spectrum)); 
	

  
	// Open the sample spectrum data	
	sampleFile = fopen("sample.dat","rb");
	
	// Read in the contents of the sample file
  // Fastnar i den här loopen och kommer inte vidare...----------------------------------------------------
	//while(fscanf(sampleFile, "%lf", &spectrum[i]) != EOF) // %lf tells fscanf to read a float
	//{
	//	i++;
  //  Serial.print("-------------------------------------------------------------");
	//}

  int k = (int) sizeof(spectrum);
  Serial.println("testteststestest");
//  for(int p = 0; p < 1000; p++){

  //  fscanf(sampleFile, "%f", &spectrum[p]);
  //  Serial.print(spectrum[p]);

  //}

  for(int p = 0; p < 1000; p++){
    //Serial.print(spectrum[p]);
    //delay(500);
    spectrum[p] = rand() % 50;
  }

	// Close the sample file
	fclose(sampleFile);

	// Compute the first 13 coefficients
	for(coeff = 0; coeff < 13; coeff++)
	{
		mfcc_result = GetCoefficient(spectrum, 16000, 40, 256, coeff);
		//String out = ("%i %f\n", coeff, mfcc_result);
    printfloat(coeff, 10);

    float f = (float) mfcc_result;

    //printfloat(mfcc_result, 100000);
    Serial.println(f);
    delay(1000);
	}
	getchar();

}
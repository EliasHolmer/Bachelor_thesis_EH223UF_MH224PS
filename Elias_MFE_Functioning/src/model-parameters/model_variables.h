#ifndef _EI_CLASSIFIER_MODEL_VARIABLES_H_
#define _EI_CLASSIFIER_MODEL_VARIABLES_H_

#include <stdint.h>
#include <C:\Users\Elias\Documents\PlatformIO\Projects\220430-220148-nano33ble\src\model-parameters\model_metadata.h>

const char* ei_classifier_inferencing_categories[] = { "Bee", "notBee" };

uint8_t ei_dsp_config_52_axes[] = { 0 };
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
    -72
};

#endif // _EI_CLASSIFIER_MODEL_METADATA_H_

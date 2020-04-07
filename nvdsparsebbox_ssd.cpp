#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))


extern "C"
bool NvDsInferParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomSSD (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  static int detectionOutLayerIndex = -1;
  static int keepCountLayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;
  static const int NUM_CLASSES_SSD = 2;

  if (detectionOutLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      //strcmp NMS default
      if (strcmp(outputLayersInfo[i].layerName, "detection_out") == 0) {
        detectionOutLayerIndex = i;
        break;
      }
    }
    if (detectionOutLayerIndex == -1) {
    std::cerr << "Could not find detection_out layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (keepCountLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      //strcmp NMS_1 default
      if (strcmp(outputLayersInfo[i].layerName, "keep_count") == 0) {
        keepCountLayerIndex = i;
        break;
      }
    }
    if (keepCountLayerIndex == -1) {
    std::cerr << "Could not find keep_count layer buffer while parsing" << std::endl;
    return false;
    }
  }

  if (!classMismatchWarn) {
    if (NUM_CLASSES_SSD !=
        detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        NUM_CLASSES_SSD << std::endl;
    }
    classMismatchWarn = true;
  }

  numClassesToParse = MIN (NUM_CLASSES_SSD, detectionParams.numClassesConfigured);

  int keep_count = *((int *) outputLayersInfo[keepCountLayerIndex].buffer);
  float *detection_out = (float *) outputLayersInfo[detectionOutLayerIndex].buffer;


  int object_to_sent = 0;
  for (int i = 0; i < keep_count; ++i)
  {
    float* det = detection_out + i * 7;
    int classId = det[1];

    if (classId >= numClassesToParse)
      continue;

    float threshold = detectionParams.perClassThreshold[classId];
    if (det[2] < threshold)
      continue;

    unsigned int rectx1, recty1, rectx2, recty2;
    NvDsInferObjectDetectionInfo object;

    rectx1 = det[3] * networkInfo.width;
    recty1 = det[4] * networkInfo.height;
    rectx2 = det[5] * networkInfo.width;
    recty2 = det[6] * networkInfo.height;

    object.classId = classId;
    object.detectionConfidence = det[2];

    /* Clip object box co-ordinates to network resolution */
    object.left = CLIP(rectx1, 0, networkInfo.width - 1);
    object.top = CLIP(recty1, 0, networkInfo.height - 1);
    object.width = CLIP(rectx2, 0, networkInfo.width - 1) -
      object.left + 1;
    object.height = CLIP(recty2, 0, networkInfo.height - 1) -
      object.top + 1;

    objectList.push_back(object);
  }
  std::cout << "| Object per frame: " << object_to_sent << std::endl;
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSD);

//
//  ShaderTypes.h
//  Raytracing Shared
//
//  Created by Jaap Wijnen on 21/11/2021.
//

//
//  Header containing types and enum constants shared between Metal shaders and
//  Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name)                                                  \
  enum _name : _type _name;                                                    \
  enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_LIGHT 2

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT)
#define RAY_MASK_SHADOW GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY

typedef NS_ENUM(NSInteger, BufferIndex) {
  BufferIndexUniforms = 0,
  BufferIndexInstanceAccelerationStructure = 1,
  BufferIndexRandom = 2,
  BufferIndexVertexColor = 3,
  BufferIndexVertexNormals = 4,
  BufferIndexResources = 5,
  BufferIndexLights = 6,
  BufferIndexInstances = 7,
  BufferIndexAccelerationStructure = 8,
  BufferIndexInstanceDescriptors = 9,

  // Skinning
  BufferIndexRestPositions = 10,
  BufferIndexRestNormals = 11,
  BufferIndexJointIndices = 12,
  BufferIndexJointWeights = 13,
  BufferIndexJointMatrices = 14,
  BufferIndexSkinnedPositions = 15,
  BufferIndexSkinnedNormals = 16,

  // Previous frame instance descriptors (motion vectors)
  BufferIndexPreviousInstanceDescriptors = 17
};

typedef NS_ENUM(NSInteger, TextureIndex) {
  TextureIndexAccumulation = 0,
  TextureIndexPreviousAccumulation = 1,
  TextureIndexRandom = 2,
  TextureIndexDepth = 3,
  TextureIndexMotion = 4,
  TextureIndexDiffuseAlbedo = 5,
  TextureIndexSpecularAlbedo = 6,
  TextureIndexNormal = 7,
  TextureIndexRoughness = 8
};

typedef NS_ENUM(NSInteger, VertexAttribute) {
  VertexAttributePosition = 0,
  VertexAttributeTexcoord = 1,
  VertexAttributeNormal = 2,
  VertexAttributeJointIndices = 3,
  VertexAttributeJointWeights = 4
};

struct Camera {
  vector_float3 position;
  vector_float3 right;
  vector_float3 up;
  vector_float3 forward;
};

typedef NS_ENUM(NSInteger, LightType) {
  LightTypeUnused = 0,
  LightTypeSunlight = 1,
  LightTypeSpotlight = 2,
  LightTypePointlight = 3,
  LightTypeAreaLight = 4
};

struct Light {
  LightType type;
  vector_float3 position;
  vector_float3 color;
  // area light
  vector_float3 forward;
  vector_float3 right;
  vector_float3 up;
  // spot light
  float coneAngle;
  vector_float3 direction;
};

struct Uniforms {
  int width;
  int height;
  int blocksWide;
  unsigned int frameIndex;
  int lightCount;
  int samplesPerPixel;
  int maxBounces;
  struct Camera camera;
  struct Camera previousCamera;
  int debugTextureMode;
  float accumulationWeight;
  int enableDenoiseGBuffer;
  int shadingMode;
  int enableMotionAdaptiveAccumulation;
  float motionAccumulationMinWeight;
  float motionAccumulationLowThresholdPixels;
  float motionAccumulationHighThresholdPixels;
};

typedef NS_ENUM(NSInteger, ShadingMode) {
  ShadingModePBR = 0,
  ShadingModeLegacy = 1
};

struct Material {
  vector_float3 baseColor;
  vector_float3 specular;
  vector_float3 emission;
  float specularExponent;
  float refractionIndex;
  unsigned int textureFlags;
};

#define MATERIAL_TEXTURE_BASECOLOR (1u << 0)
#define MATERIAL_TEXTURE_NORMAL (1u << 1)
#define MATERIAL_TEXTURE_ROUGHNESS (1u << 2)
#define MATERIAL_TEXTURE_METALLIC (1u << 3)
#define MATERIAL_TEXTURE_AO (1u << 4)
#define MATERIAL_TEXTURE_EMISSION (1u << 5)
#define MATERIAL_TEXTURE_OPACITY (1u << 6)

#ifndef ENABLE_AO
#define ENABLE_AO 0
#endif

typedef NS_ENUM(NSInteger, DebugTextureMode) {
  DebugTextureModeNone = 0,
  DebugTextureModeBaseColor = 1,
  DebugTextureModeNormal = 2,
  DebugTextureModeRoughness = 3,
  DebugTextureModeMetallic = 4,
  DebugTextureModeAO = 5,
  DebugTextureModeEmission = 6,
  DebugTextureModeMotion = 7
};

#endif /* ShaderTypes_h */

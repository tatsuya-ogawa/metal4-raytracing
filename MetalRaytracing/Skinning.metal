#include <metal_stdlib>
using namespace metal;

#import "ShaderTypes.h"

static inline void accumulateSkinningContribution(ushort jointIndex,
                                                  float weight,
                                                  float3 position,
                                                  float3 normal,
                                                  constant float4x4 *jointMatrices,
                                                  uint jointCount,
                                                  thread float4 &skinnedPos,
                                                  thread float3 &skinnedNrm)
{
    if (weight <= 0.0 || jointIndex >= jointCount) {
        return;
    }

    const float4x4 jointMatrix = jointMatrices[jointIndex];
    skinnedPos += weight * (jointMatrix * float4(position, 1.0));
    skinnedNrm += weight * (jointMatrix * float4(normal, 0.0)).xyz;
}

// Kernel to perform linear blend skinning on the GPU
kernel void skinningKernel(uint vertexID [[thread_position_in_grid]],
                           constant float3 *restPositions [[buffer(BufferIndexRestPositions)]],
                           constant float3 *restNormals [[buffer(BufferIndexRestNormals)]],
                           constant ushort4 *jointIndices [[buffer(BufferIndexJointIndices)]],
                           constant float4 *jointWeights [[buffer(BufferIndexJointWeights)]],
                           constant float4x4 *jointMatrices [[buffer(BufferIndexJointMatrices)]],
                           device float3 *skinnedPositions [[buffer(BufferIndexSkinnedPositions)]],
                           device float3 *skinnedNormals [[buffer(BufferIndexSkinnedNormals)]],
                           constant SkinningUniforms &uniforms [[buffer(BufferIndexUniforms)]])
{
    if (vertexID >= uniforms.vertexCount) {
        return;
    }

    float3 position = restPositions[vertexID];
    float3 normal = restNormals[vertexID];
    ushort4 indices = jointIndices[vertexID];
    float4 weights = jointWeights[vertexID];
    
    // Only fallback to first joint if weights are effectively zero
    // Do NOT normalize - use weights as authored in the asset
    float weightSum = weights.x + weights.y + weights.z + weights.w;
    if (weightSum < 0.0001) {
        weights = float4(1.0, 0.0, 0.0, 0.0);
    }

    float4 skinnedPos = float4(0.0);
    float3 skinnedNrm = float3(0.0);

    // Skip invalid palette indices instead of reading outside the joint palette.
    accumulateSkinningContribution(indices.x, weights.x, position, normal, jointMatrices, uniforms.jointCount, skinnedPos, skinnedNrm);
    accumulateSkinningContribution(indices.y, weights.y, position, normal, jointMatrices, uniforms.jointCount, skinnedPos, skinnedNrm);
    accumulateSkinningContribution(indices.z, weights.z, position, normal, jointMatrices, uniforms.jointCount, skinnedPos, skinnedNrm);
    accumulateSkinningContribution(indices.w, weights.w, position, normal, jointMatrices, uniforms.jointCount, skinnedPos, skinnedNrm);

    skinnedPositions[vertexID] = skinnedPos.xyz;
    skinnedNormals[vertexID] = skinnedNrm;
}

#include <metal_stdlib>
using namespace metal;

#import "ShaderTypes.h"

// Kernel to perform linear blend skinning on the GPU
kernel void skinningKernel(uint vertexID [[thread_position_in_grid]],
                           constant float3 *restPositions [[buffer(BufferIndexRestPositions)]],
                           constant float3 *restNormals [[buffer(BufferIndexRestNormals)]],
                           constant ushort4 *jointIndices [[buffer(BufferIndexJointIndices)]],
                           constant float4 *jointWeights [[buffer(BufferIndexJointWeights)]],
                           constant float4x4 *jointMatrices [[buffer(BufferIndexJointMatrices)]],
                           device float3 *skinnedPositions [[buffer(BufferIndexSkinnedPositions)]],
                           device float3 *skinnedNormals [[buffer(BufferIndexSkinnedNormals)]],
                           constant uint &vertexCount [[buffer(BufferIndexUniforms)]])
{
    if (vertexID >= vertexCount) {
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

    // Unroll manually for 4 weights
    skinnedPos += weights.x * (jointMatrices[indices.x] * float4(position, 1.0));
    skinnedPos += weights.y * (jointMatrices[indices.y] * float4(position, 1.0));
    skinnedPos += weights.z * (jointMatrices[indices.z] * float4(position, 1.0));
    skinnedPos += weights.w * (jointMatrices[indices.w] * float4(position, 1.0));

    skinnedNrm += weights.x * (jointMatrices[indices.x] * float4(normal, 0.0)).xyz;
    skinnedNrm += weights.y * (jointMatrices[indices.y] * float4(normal, 0.0)).xyz;
    skinnedNrm += weights.z * (jointMatrices[indices.z] * float4(normal, 0.0)).xyz;
    skinnedNrm += weights.w * (jointMatrices[indices.w] * float4(normal, 0.0)).xyz;

    skinnedPositions[vertexID] = skinnedPos.xyz;
    skinnedNormals[vertexID] = skinnedNrm;
}

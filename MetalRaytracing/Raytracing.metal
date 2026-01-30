//
//  Shaders.metal
//  Raytracing Shared
//
//  Created by Jaap Wijnen on 21/11/2021.
//  Updated by Tatsuya Ogawa on 26/01/2026
//

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;
using namespace raytracing;

struct TriangleResources {
    device float3 *vertexNormals;
    device float3 *vertexColors;
};

constant int resourcesStride [[function_constant(0)]];
constant int maxSubmeshes [[function_constant(1)]];

constant short constNumbers[64] = { 41, 38, 64, 45, 35, 59, 44, 11, 54, 29, 47, 26, 19, 22, 24, 7, 1, 23, 50, 9, 5, 52, 4, 56, 39, 0, 55, 25, 53, 16, 14, 13, 18, 15, 40, 60, 63, 21, 51, 30, 32, 10, 12, 33, 36, 6, 43, 57, 42, 62, 20, 28, 31, 17, 46, 34, 37, 3, 61, 58, 2, 27, 49, 8 };

constant short primes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541
};

// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base. The Halton sequence is a "low discrepency" sequence: the values appear
// random but are more evenly distributed than a purely random sequence. Each random
// value used to render the image should use a different independent dimension 'd',
// and each sample (frame) should use a different index 'i'. To decorrelate each
// pixel, you can apply a random offset to 'i'.
float halton(int i, short d) {
    short b = primes[d];

    float f = 1.0f;
    float invB = 1.0f / b;

    float r = 0;

    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }

    return r;
}

// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
inline T interpolateVertexAttribute(device T *attributes, intersector<triangle_data, instancing>::result_type intersection, device int *vertexIndices) {
    float3 uvw;
    uvw.xy = intersection.triangle_barycentric_coord;
    uvw.z = 1.0 - uvw.x - uvw.y;
    unsigned int triangleIndex = intersection.primitive_id;
    unsigned int index1 = vertexIndices[triangleIndex * 3 + 1];
    unsigned int index2 = vertexIndices[triangleIndex * 3 + 2];
    unsigned int index3 = vertexIndices[triangleIndex * 3 + 0];
    T T0 = attributes[index1];
    T T1 = attributes[index2];
    T T2 = attributes[index3];
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

// Uses the inversion method to map two uniformly random numbers to a three dimensional
// unit hemisphere where the probability of a given sample is proportional to the cosine
// of the angle between the sample direction and the "up" direction (0, 1, 0)
inline float3 sampleCosineWeightedHemisphere(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;
    
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    
    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
}

// Maps two uniformly random numbers to the surface of a two-dimensional area light
// source and returns the direction to this point, the amount of light which travels
// between the intersection point and the sample point on the light source, as well
// as the distance between these two points.
inline void sampleAreaLight(device Light & light,
                            float2 u,
                            float3 position,
                            thread float3 & lightDirection,
                            thread float3 & lightColor,
                            thread float & lightDistance)
{
    // Map to -1..1
    u = u * 2.0f - 1.0f;
    
    // Transform into light's coordinate system
    float3 samplePosition = light.position +
    light.right * u.x +
    light.up * u.y;
    
    // Compute vector from sample point on light source to intersection point
    lightDirection = samplePosition - position;
    
    lightDistance = length(lightDirection);
    
    float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
    
    // Normalize the light direction
    lightDirection *= inverseLightDistance;
    
    // Start with the light's color
    lightColor = light.color;
    
    // Light falls off with the inverse square of the distance to the intersection point
    lightColor *= (inverseLightDistance * inverseLightDistance);
    
    // Light also falls off with the cosine of angle between the intersection point and
    // the light source
    lightColor *= saturate(dot(-lightDirection, light.forward));
}

// Aligns a direction on the unit hemisphere such that the hemisphere's "up" direction
// (0, 1, 0) maps to the given surface normal direction
inline float3 alignHemisphereWithNormal(float3 sample, float3 normal) {
    // Set the "up" vector to the normal
    float3 up = normal;
    
    // Find an arbitrary direction perpendicular to the normal. This will become the
    // "right" vector.
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));
    
    // Find a third vector perpendicular to the previous two. This will be the
    // "forward" vector.
    float3 forward = cross(right, up);
    
    // Map the direction on the unit hemisphere to the coordinate system aligned
    // with the normal.
    return sample.x * right + sample.y * up + sample.z * forward;
}

inline float distributionGGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(M_PI_F * denom * denom, 1e-7f);
}

inline float geometrySchlickGGX(float NdotV, float k) {
    return NdotV / max(NdotV * (1.0f - k) + k, 1e-7f);
}

inline float geometrySmith(float NdotV, float NdotL, float k) {
    return geometrySchlickGGX(NdotV, k) * geometrySchlickGGX(NdotL, k);
}

inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

struct Resource
{
    device float3 *positions [[id(0)]];
    device float3 *previousPositions [[id(1)]];
    device float3 *normals [[id(2)]];
    device int *indices [[id(3)]];
    device Material *material [[id(4)]];
    device float2 *uvs [[id(5)]];
    texture2d<float> baseColorMap [[id(6)]];
    texture2d<float> normalMap [[id(7)]];
    texture2d<float> roughnessMap [[id(8)]];
    texture2d<float> metallicMap [[id(9)]];
    texture2d<float> aoMap [[id(10)]];
    texture2d<float> opacityMap [[id(11)]];
    texture2d<float> emissionMap [[id(12)]];
};

inline bool computeTangentBasis(device float3 *positions,
                                device float2 *uvs,
                                intersector<triangle_data, instancing>::result_type intersection,
                                device int *vertexIndices,
                                thread float3 &tangent,
                                thread float3 &bitangent)
{
    unsigned int triangleIndex = intersection.primitive_id;
    unsigned int index1 = vertexIndices[triangleIndex * 3 + 1];
    unsigned int index2 = vertexIndices[triangleIndex * 3 + 2];
    unsigned int index3 = vertexIndices[triangleIndex * 3 + 0];
    
    float3 p0 = positions[index1];
    float3 p1 = positions[index2];
    float3 p2 = positions[index3];
    
    float2 uv0 = uvs[index1];
    float2 uv1 = uvs[index2];
    float2 uv2 = uvs[index3];
    
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float2 dUV1 = uv1 - uv0;
    float2 dUV2 = uv2 - uv0;
    
    float denom = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
    if (fabs(denom) < 1e-8f) {
        return false;
    }
    float r = 1.0f / denom;
    tangent = (e1 * dUV2.y - e2 * dUV1.y) * r;
    bitangent = (e2 * dUV1.x - e1 * dUV2.x) * r;
    return (length(tangent) > 1e-8f) && (length(bitangent) > 1e-8f);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void raytracingKernel(uint2 tid [[thread_position_in_grid]],
                             constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                             acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]],
                             device Resource *resources [[ buffer(BufferIndexResources) ]],
                             device MTLIndirectAccelerationStructureInstanceDescriptor *instances [[ buffer(BufferIndexInstanceDescriptors) ]],
                             device MTLIndirectAccelerationStructureInstanceDescriptor *prevInstances [[ buffer(BufferIndexPreviousInstanceDescriptors) ]],
                             device Light *lights [[ buffer(BufferIndexLights) ]],
                             texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]],
                             texture2d<float, access::read> prevTex [[ texture(TextureIndexAccumulation) ]],
                             texture2d<float, access::write> dstTex [[ texture(TextureIndexPreviousAccumulation) ]],
                             texture2d<float, access::write> depthTex [[ texture(TextureIndexDepth) ]],
                             texture2d<float, access::read_write> motionTex [[ texture(TextureIndexMotion) ]],
                             texture2d<float, access::write> diffuseAlbedoTex [[ texture(TextureIndexDiffuseAlbedo) ]],
                             texture2d<float, access::write> specularAlbedoTex [[ texture(TextureIndexSpecularAlbedo) ]],
                             texture2d<float, access::write> normalTex [[ texture(TextureIndexNormal) ]],
                             texture2d<float, access::write> roughnessTex [[ texture(TextureIndexRoughness) ]])
{
    // The sample aligns the thread count to the threadgroup size. which means the thread count
    // may be different than the bounds of the texture. Test to make sure this thread
    // is referencing a pixel within the bounds of the texture.
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        float2 pixel = (float2)tid;
        
        // Apply a random offset to the random number index to decorrelate pixels.
        unsigned int offset = randomTexture.read(tid).x;
        
        float3 totalColor = float3(0.0f);
        float2 prevMotion = motionTex.read(tid).xy;
        
        // For temporal scaler: track first hit for depth and motion
        float primaryDepth = 0.0f;
        float2 motionVector = float2(0.0f);
        bool hadPrimaryHit = false;
        
        float4 outDiffuseAlbedo = float4(0.0f);
        float4 outSpecularAlbedo = float4(0.0f);
        float4 outNormal = float4(0.0f);
        float4 outRoughness = float4(0.0f);
        bool wroteGBuffer = false;
        
        // Multiple samples per pixel
        for (int sampleIndex = 0; sampleIndex < uniforms.samplesPerPixel; sampleIndex++) {
            int frameOffset = uniforms.frameIndex * uniforms.samplesPerPixel + sampleIndex;
            
            // Add a random offset to the pixel coordinates for antialiasing.
            float2 r = float2(halton(offset + frameOffset, 0),
                              halton(offset + frameOffset, 1));
            float2 samplePixel = (float2)tid + r;
        
            // Map pixel coordinates to -1..1.
            float2 uv = samplePixel / float2(uniforms.width, uniforms.height);
            uv = uv * 2.0 - 1.0;
        
        constant Camera & camera = uniforms.camera;
        
        ray ray;
        // Rays start at the camera position.
        ray.origin = camera.position;
        // Map normalized pixel coordinates into camera's coordinate system.
        ray.direction = normalize(uv.x * camera.right +
                                  uv.y * camera.up +
                                  camera.forward);
        // Don't limit intersection distance.
        ray.max_distance = INFINITY;
        ray.min_distance = 0;
        
        // Start with a fully white color. The kernel scales the light each time the
        // ray bounces off of a surface, based on how much of each light component
        // the surface absorbs.
        float3 color = float3(1.0f, 1.0f, 1.0f);
        float3 accumulatedColor = float3(0.0f, 0.0f, 0.0f);
        
        // Create an intersector to test for intersection between the ray and the geometry in the scene.
        intersector<triangle_data, instancing> i;
                
        i.assume_geometry_type(geometry_type::triangle);
        i.force_opacity(forced_opacity::opaque);
        
        typename intersector<triangle_data, instancing>::result_type intersection;
                
        int bounce = 0;
        int step = 0;
        int transparencyPasses = 0;
        while (bounce < uniforms.maxBounces) {
            // Get the closest intersection, not the first intersection. This is the default, but
            // the sample adjusts this property below when it casts shadow rays.
            i.accept_any_intersection(false);
            
            // Check for intersection between the ray and the acceleration structure.
            //intersection = i.intersect(ray, accelerationStructure, bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY);
            intersection = i.intersect(ray, accelerationStructure);
            
            // Stop if the ray didn't hit anything and has bounced out of the scene.
            if (intersection.type == intersection_type::none)
                break;
            
            int instanceIndex = intersection.instance_id;
            // only single instances currently
            int geometryIndex2 = intersection.geometry_id;
            
            // The ray hit something. Look up the transformation matrix for this instance.
            float4x4 objectToWorldSpaceTransform(1.0f);

            for (int column = 0; column < 4; column++)
                for (int row = 0; row < 3; row++)
                    objectToWorldSpaceTransform[column][row] = instances[instanceIndex].transformationMatrix[column][row];
            
            // Compute intersection point in world space.
            float3 worldSpaceIntersectionPoint = ray.origin + ray.direction * intersection.distance;
            int maxSubmeshes2 = maxSubmeshes;
            int resourceIndex = instanceIndex * maxSubmeshes2 + geometryIndex2;
            Resource resource = resources[resourceIndex];
            
            // Store depth and motion for first bounce (primary ray)
            if (bounce == 0 && sampleIndex == 0) {
                // Depth from camera
                primaryDepth = intersection.distance;
                
                // Compute object-space position and previous object-space position
                float3 objectSpacePos = interpolateVertexAttribute(resource.positions, intersection, resource.indices);
                float3 prevObjectSpacePos = interpolateVertexAttribute(resource.previousPositions, intersection, resource.indices);
                
                // Current world position
                float3 worldPos = (objectToWorldSpaceTransform * float4(objectSpacePos, 1.0)).xyz;
                
                // Previous world position (previous instance transform + previous skinned pos)
                float4x4 prevObjectToWorldSpaceTransform(1.0f);
                for (int column = 0; column < 4; column++) {
                    for (int row = 0; row < 3; row++) {
                        prevObjectToWorldSpaceTransform[column][row] = prevInstances[instanceIndex].transformationMatrix[column][row];
                    }
                }
                float3 prevWorldPos = (prevObjectToWorldSpaceTransform * float4(prevObjectSpacePos, 1.0)).xyz;
                
                // Project current position
                constant Camera & camera = uniforms.camera;
                float3 viewPos = worldPos - camera.position;
                float2 screenPos;
                screenPos.x = dot(viewPos, camera.right);
                screenPos.y = dot(viewPos, camera.up);
                float depth = dot(viewPos, camera.forward);
                screenPos /= max(depth, 0.001f);
                
                // Project previous position
                constant Camera & prevCamera = uniforms.previousCamera;
                float3 prevViewPos = prevWorldPos - prevCamera.position;
                float2 prevScreenPos;
                prevScreenPos.x = dot(prevViewPos, prevCamera.right);
                prevScreenPos.y = dot(prevViewPos, prevCamera.up);
                float prevDepth = dot(prevViewPos, prevCamera.forward);
                prevScreenPos /= max(prevDepth, 0.001f);
                
                // Motion vector in screen space
                motionVector = screenPos - prevScreenPos;
                hadPrimaryHit = true;
            }
            
            float3 objectSpaceSurfaceNormal = interpolateVertexAttribute(resource.normals, intersection, resource.indices);
            float3 worldSpaceSurfaceNormal = (objectToWorldSpaceTransform * float4(objectSpaceSurfaceNormal, 0)).xyz;
            worldSpaceSurfaceNormal = normalize(worldSpaceSurfaceNormal);
            // for invalid normal model
            if (length(objectSpaceSurfaceNormal) < 1e-10f) {
                worldSpaceSurfaceNormal = -ray.direction;
            }
            
            float3 albedo = resource.material->baseColor;
            uint textureFlags = resource.material->textureFlags;
            bool hasBaseColorMap = (textureFlags & MATERIAL_TEXTURE_BASECOLOR) != 0;
            bool hasNormalMap = (textureFlags & MATERIAL_TEXTURE_NORMAL) != 0;
            bool hasRoughnessMap = (textureFlags & MATERIAL_TEXTURE_ROUGHNESS) != 0;
            bool hasMetallicMap = (textureFlags & MATERIAL_TEXTURE_METALLIC) != 0;
#if ENABLE_AO
            bool hasAOMap = (textureFlags & MATERIAL_TEXTURE_AO) != 0;
#else
            bool hasAOMap = false;
#endif
            bool hasOpacityMap = (textureFlags & MATERIAL_TEXTURE_OPACITY) != 0;
            bool hasEmissionMap = (textureFlags & MATERIAL_TEXTURE_EMISSION) != 0;
            
            float2 texCoord = float2(0.0f);
            if (hasBaseColorMap || hasNormalMap || hasRoughnessMap || hasMetallicMap || hasAOMap || hasOpacityMap || hasEmissionMap) {
                texCoord = interpolateVertexAttribute(resource.uvs, intersection, resource.indices);
                // Flip Y coordinate for USDZ textures (OpenGL style UV -> Metal style)
                texCoord.y = 1.0f - texCoord.y;
            }
            
            // Texture Sampler
            constexpr sampler sampler(min_filter::linear, mag_filter::linear, mip_filter::linear, address::repeat);
            
            // Sample Textures
            float4 baseColorSample = float4(1.0f);
            if (hasBaseColorMap) {
                 baseColorSample = resource.baseColorMap.sample(sampler, texCoord);
                 albedo *= baseColorSample.rgb;
            }
            
            // New PBR Textures
            float roughness = 1.0f;
            if (hasRoughnessMap) {
                roughness = resource.roughnessMap.sample(sampler, texCoord).x;
            }
            
            float metallic = 0.0f;
            if (hasMetallicMap) {
                metallic = resource.metallicMap.sample(sampler, texCoord).x;
            }
            
            float ao = 1.0f;
#if ENABLE_AO
            if (hasAOMap) {
                ao = resource.aoMap.sample(sampler, texCoord).x;
            }
#endif

            float opacity = 1.0f;
            if (hasOpacityMap) {
                opacity = resource.opacityMap.sample(sampler, texCoord).x;
            }
            
            float3 emission = resource.material->emission;
            if (hasEmissionMap) {
                emission = resource.emissionMap.sample(sampler, texCoord).xyz;
            }
            
            // Debug Visualization
            if (uniforms.debugTextureMode != DebugTextureModeNone) {
                float3 debugColor = float3(0.0f);
                if (uniforms.debugTextureMode == DebugTextureModeBaseColor) {
                    debugColor = hasBaseColorMap ? baseColorSample.rgb : float3(1.0f, 0.0f, 1.0f); // Magenta if missing
                } else if (uniforms.debugTextureMode == DebugTextureModeNormal) {
                    if (hasNormalMap) {
                         debugColor = resource.normalMap.sample(sampler, texCoord).xyz;
                    } else {
                         debugColor = float3(0.5f, 0.5f, 1.0f); // Flat normal
                    }
                } else if (uniforms.debugTextureMode == DebugTextureModeRoughness) {
                    debugColor = float3(roughness);
                } else if (uniforms.debugTextureMode == DebugTextureModeMetallic) {
                    debugColor = float3(metallic);
                } else if (uniforms.debugTextureMode == DebugTextureModeAO) {
#if ENABLE_AO
                    debugColor = float3(ao);
#else
                    debugColor = float3(1.0f, 0.0f, 1.0f); // AO disabled
#endif
                } else if (uniforms.debugTextureMode == DebugTextureModeEmission) {
                    debugColor = emission;
                } else if (uniforms.debugTextureMode == DebugTextureModeMotion) {
                    float2 motionForViz = hadPrimaryHit ? motionVector : prevMotion;
                    float rightScale = max(length(uniforms.camera.right), 1e-5f);
                    float upScale = max(length(uniforms.camera.up), 1e-5f);
                    float2 motionPixels = float2(
                        motionForViz.x * (float(uniforms.width) / (2.0f * rightScale)),
                        motionForViz.y * (float(uniforms.height) / (2.0f * upScale))
                    );
                    float2 scaled = clamp(motionPixels * 0.05f, -1.0f, 1.0f);
                    float mag = clamp(length(motionPixels) * 0.1f, 0.0f, 1.0f);
                    debugColor = float3(scaled.x * 0.5f + 0.5f, scaled.y * 0.5f + 0.5f, mag);
                }
                accumulatedColor = debugColor;
                break; // Exit bounce loop for debug view
            }
            
            float3 shadingNormal = worldSpaceSurfaceNormal;
            if (hasNormalMap) {
                float3 tangent;
                float3 bitangent;
                if (computeTangentBasis(resource.positions, resource.uvs, intersection, resource.indices, tangent, bitangent)) {
                    float3 worldT = (objectToWorldSpaceTransform * float4(tangent, 0)).xyz;
                    worldT = normalize(worldT - worldSpaceSurfaceNormal * dot(worldT, worldSpaceSurfaceNormal));
                    float3 worldBOrtho = normalize(cross(worldSpaceSurfaceNormal, worldT));
                    
                    float3 nMap = resource.normalMap.sample(sampler, texCoord).xyz * 2.0f - 1.0f;
                    shadingNormal = normalize(nMap.x * worldT + nMap.y * worldBOrtho + nMap.z * worldSpaceSurfaceNormal);
                }
            }

            if (uniforms.enableDenoiseGBuffer != 0 && !wroteGBuffer && sampleIndex == 0) {
                float roughnessForOutput = clamp(roughness, 0.0f, 1.0f);
                float3 diffuseAlbedo = albedo * (1.0f - metallic);
                float3 specularAlbedo = mix(float3(0.04f), albedo, metallic);
                outDiffuseAlbedo = float4(diffuseAlbedo, 1.0f);
                outSpecularAlbedo = float4(specularAlbedo, 1.0f);
                outNormal = float4(shadingNormal * 0.5f + 0.5f, 1.0f);
                outRoughness = float4(roughnessForOutput, 0.0f, 0.0f, 1.0f);
                wroteGBuffer = true;
            }

            float clampedOpacity = clamp(opacity, 0.0f, 1.0f);
            float ior = max(resource.material->refractionIndex, 1.0f);
            bool consumeBounce = true;
            bool skipLighting = false;
            if (clampedOpacity < 0.999f || ior > 1.01f) {
                float3 N = shadingNormal;
                float3 I = ray.direction;
                float cosi = clamp(dot(-I, N), -1.0f, 1.0f);
                float etaI = 1.0f;
                float etaT = ior;
                if (cosi < 0.0f) {
                    cosi = -cosi;
                    N = -N;
                    float tmp = etaI; etaI = etaT; etaT = tmp;
                }
                float eta = etaI / etaT;
                float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
                
                float f0 = (etaT - etaI) / (etaT + etaI);
                f0 = f0 * f0;
                float F = f0 + (1.0f - f0) * pow(clamp(1.0f - cosi, 0.0f, 1.0f), 5.0f);
                
                float transmission = 1.0f - clampedOpacity;
                float reflectWeight = F;
                float refractWeight = (1.0f - F) * transmission;
                float totalWeight = max(reflectWeight + refractWeight, 1e-4f);
                float reflectProb = reflectWeight / totalWeight;
                
                float choice = halton(offset + frameOffset, 2 + step * 6 + 5);
                
                if (k < 0.0f || choice < reflectProb) {
                    float3 reflectDir = normalize(I - 2.0f * dot(I, N) * N);
                    ray.origin = worldSpaceIntersectionPoint + reflectDir * 1e-3f;
                    ray.direction = reflectDir;
                    color *= totalWeight;
                } else {
                    float cosT = sqrt(max(k, 0.0f));
                    float3 refractDir = normalize(eta * I + (eta * cosi - cosT) * N);
                    ray.origin = worldSpaceIntersectionPoint + refractDir * 1e-3f;
                    ray.direction = refractDir;
                    color *= totalWeight * albedo;
                    consumeBounce = false;
                }
                skipLighting = true;
            }

            if (skipLighting) {
                step++;
                if (consumeBounce) {
                    bounce++;
                    transparencyPasses = 0;
                } else {
                    transparencyPasses++;
                    if (transparencyPasses > uniforms.maxBounces) {
                        bounce++;
                        transparencyPasses = 0;
                    }
                }
                continue;
            }

            float perceptualRoughness = clamp(roughness, 0.04f, 1.0f);
            float alpha = perceptualRoughness * perceptualRoughness;
            float3 diffuseColor = albedo;
            float3 F0 = mix(float3(0.04f), albedo, metallic);
            float3 V = normalize(-ray.direction);

            // Emission (throughput-scaled)
            accumulatedColor += color * emission;
            
            // Choose a random light source to sample.
            float lightSample = halton(offset + frameOffset, 2 + step * 6 + 0);
            int lightIndex = min((int)(lightSample * uniforms.lightCount), uniforms.lightCount - 1);
            
            device Light &light = lights[lightIndex];
            
            float3 worldSpaceLightDirection;
            float lightDistance;
            float3 lightColor;
            
            if (light.type == LightTypeAreaLight) {
                
                // Choose a random point to sample on the light source.
                r = float2(halton(offset + frameOffset, 2 + step * 6 + 1),
                                  halton(offset + frameOffset, 2 + step * 6 + 2));

                // Sample the lighting between the intersection point and the point on the area light.
                sampleAreaLight(light, r, worldSpaceIntersectionPoint, worldSpaceLightDirection,
                                lightColor, lightDistance);
                
                
            } else if (light.type == LightTypeSpotlight) {
                // Compute vector from sample point on light source to intersection point
                worldSpaceLightDirection = light.position - worldSpaceIntersectionPoint;
                        
                lightDistance = length(worldSpaceLightDirection);
                
                float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
                
                // Normalize the light direction
                worldSpaceLightDirection *= inverseLightDistance;
                
//                // Start with the light's color
//                lightColor = light.color;
//
//                // Light falls off with the inverse square of the distance to the intersection point
//                lightColor *= (inverseLightDistance * inverseLightDistance);
                
                lightColor = 0.0;
                
                float3 coneDirection = normalize(light.direction);
                float spotResult = dot(-worldSpaceLightDirection, coneDirection);
                
                if (spotResult > cos(light.coneAngle)) {
                    lightColor = light.color * inverseLightDistance * inverseLightDistance;
                }
            } else if (light.type == LightTypePointlight) {
                worldSpaceLightDirection = light.position - worldSpaceIntersectionPoint;
                lightDistance = length(worldSpaceLightDirection);
                float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
                worldSpaceLightDirection *= inverseLightDistance;
                lightColor = light.color * inverseLightDistance * inverseLightDistance;
            } else  { // light.type == LightTypeSunlight
                worldSpaceLightDirection = -normalize(light.direction);
                lightDistance = INFINITY;
                lightColor = light.color;
            }
            
            // Scale the light color by the number of lights to compensate for the fact that
            // the sample only samples one light source at random.
            lightColor *= uniforms.lightCount;

            if (uniforms.shadingMode == ShadingModeLegacy) {
                float3 L = normalize(worldSpaceLightDirection);
                float NdotL = saturate(dot(shadingNormal, L));
                float3 legacyColor = color * albedo;

                if (length(legacyColor) < 0.001) {
                    break;
                }

                if (length(lightColor) > 0.0001 && NdotL > 0.0f) {
                    struct ray shadowRay;
                    shadowRay.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
                    shadowRay.direction = worldSpaceLightDirection;
                    shadowRay.max_distance = lightDistance - 1e-3f;

                    i.accept_any_intersection(true);
                    intersection = i.intersect(shadowRay, accelerationStructure);

                    if (intersection.type == intersection_type::none) {
                        accumulatedColor += legacyColor * lightColor * NdotL;
                    }
                }

                color = legacyColor * ao;
                if (length(color) < 0.001) {
                    break;
                }

                r = float2(halton(offset + frameOffset, 2 + step * 5 + 3),
                           halton(offset + frameOffset, 2 + step * 5 + 4));

                float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(r);
                worldSpaceSampleDirection = alignHemisphereWithNormal(worldSpaceSampleDirection, shadingNormal);

                ray.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
                ray.direction = worldSpaceSampleDirection;

                step++;
                bounce++;
                transparencyPasses = 0;
                continue;
            }
            
            if (length(lightColor) > 0.0001) {
                float3 L = normalize(worldSpaceLightDirection);
                float3 H = normalize(V + L);
                float NdotL = saturate(dot(shadingNormal, L));
                float NdotV = saturate(dot(shadingNormal, V));
                float NdotH = saturate(dot(shadingNormal, H));
                float VdotH = saturate(dot(V, H));
                
                float3 F = fresnelSchlick(VdotH, F0);
                float D = distributionGGX(NdotH, alpha);
                float k = (perceptualRoughness + 1.0f);
                k = (k * k) / 8.0f;
                float G = geometrySmith(NdotV, NdotL, k);
                
                float3 specular = (D * G) * F / max(4.0f * NdotV * NdotL, 1e-4f);
                float3 kS = F;
                float3 kD = (1.0f - kS) * (1.0f - metallic);
                float3 diffuse = kD * diffuseColor / M_PI_F;
                
                float3 direct = (diffuse + specular) * lightColor * NdotL;
            
                // Compute the shadow ray. The shadow ray checks if the sample position on the
                // light source is visible from the current intersection point.
                // If it is, the lighting contribution is added to the output image.
                struct ray shadowRay;

                // Add a small offset to the intersection point to avoid intersecting the same
                // triangle again.
                shadowRay.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;

                // Travel towards the light source.
                shadowRay.direction = worldSpaceLightDirection;

                // Don't overshoot the light source.
                shadowRay.max_distance = lightDistance - 1e-3f;

                // Shadow rays check only whether there is an object between the intersection point
                // and the light source. Tell Metal to return after finding any intersection.
                i.accept_any_intersection(true);

                /*if (useIntersectionFunctions)
                    intersection = i.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW, intersectionFunctionTable);
                else
                    intersection = i.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW);
                 */
                intersection = i.intersect(shadowRay, accelerationStructure);

                // If there was no intersection, then the light source is visible from the original
                // intersection  point. Add the light's contribution to the image.
                if (intersection.type == intersection_type::none) {
                    accumulatedColor += color * direct;
                }
            }

            // Update throughput for next bounce (diffuse only)
            // Apply AO to indirect only to avoid darkening direct lighting too much.
            color *= diffuseColor * (1.0f - metallic) * ao;
            
            // Early exit if contribution becomes negligible
            if (length(color) < 0.001) {
                break;
            }

            // Next choose a random direction to continue the path of the ray. This will
            // cause light to bounce between surfaces. The sample could apply a fair bit of math
            // to compute the fraction of light reflected by the current intersection point to the
            // previous point from the next point. However, by choosing a random direction with
            // probability proportional to the cosine (dot product) of the angle between the
            // sample direction and surface normal, the math entirely cancels out except for
            // multiplying by the surface color. This sampling strategy also reduces the amount
            // of noise in the output image.
            r = float2(halton(offset + frameOffset, 2 + step * 5 + 3),
                       halton(offset + frameOffset, 2 + step * 5 + 4));

            float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(r);
            worldSpaceSampleDirection = alignHemisphereWithNormal(worldSpaceSampleDirection, shadingNormal);

            ray.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
            ray.direction = worldSpaceSampleDirection;
            
            step++;
            bounce++;
            transparencyPasses = 0;
        }
        
        totalColor += accumulatedColor;
    } // End of samples loop
    
    // Average all samples
    totalColor /= uniforms.samplesPerPixel;
                    
    // Average this frame's sample with all of the previous frames.
    if (uniforms.frameIndex > 0) {
        float3 prevColor = prevTex.read(tid).xyz;
        float historyWeight = clamp(uniforms.accumulationWeight, 0.0f, 0.95f);
        if (uniforms.enableMotionAdaptiveAccumulation != 0) {
            float rightScale = max(length(uniforms.camera.right), 1e-5f);
            float upScale = max(length(uniforms.camera.up), 1e-5f);
            
            // Calculate motion pixels for current frame
            float2 currentMotionPixels = float2(
                motionVector.x * (float(uniforms.width) / (2.0f * rightScale)),
                motionVector.y * (float(uniforms.height) / (2.0f * upScale))
            );
            
            // Calculate motion pixels for previous frame
            float2 prevMotionPixels = float2(
                prevMotion.x * (float(uniforms.width) / (2.0f * rightScale)),
                prevMotion.y * (float(uniforms.height) / (2.0f * upScale))
            );
            
            // Use the maximum motion to determine accumulation weight.
            // This handles cases where an object stops (current is 0, prev was high)
            // or disocclusion (current is 0 (background), prev was high (foreground object))
            float motionMag = max(length(currentMotionPixels), length(prevMotionPixels));
            
            float low = max(uniforms.motionAccumulationLowThresholdPixels, 0.0f);
            float high = max(uniforms.motionAccumulationHighThresholdPixels, low + 1e-3f);
            float t = clamp((motionMag - low) / (high - low), 0.0f, 1.0f);
            float minWeight = clamp(uniforms.motionAccumulationMinWeight, 0.0f, 0.95f);
            minWeight = min(minWeight, historyWeight);
            historyWeight = mix(historyWeight, minWeight, t);
        }
        totalColor = mix(totalColor, prevColor, historyWeight);
    }

    dstTex.write(float4(totalColor, 1.0f), tid);
    
    // Write depth and motion for temporal scaler (if textures are bound)
    depthTex.write(primaryDepth, tid);
    motionTex.write(float4(motionVector, 0.0f, 0.0f), tid);
    if (uniforms.enableDenoiseGBuffer != 0) {
        diffuseAlbedoTex.write(outDiffuseAlbedo, tid);
        specularAlbedoTex.write(outSpecularAlbedo, tid);
        normalTex.write(outNormal, tid);
        roughnessTex.write(outRoughness, tid);
    }
    }
}

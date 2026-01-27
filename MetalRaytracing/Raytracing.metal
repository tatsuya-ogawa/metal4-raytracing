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

struct Resource
{
    device float3 *normals;
    device int *indices;
    device Material *material;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void raytracingKernel(uint2 tid [[thread_position_in_grid]],
                             constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                             acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]],
                             device Resource *resources [[ buffer(BufferIndexResources) ]],
                             device MTLIndirectAccelerationStructureInstanceDescriptor *instances [[ buffer(BufferIndexInstanceDescriptors) ]],
                             device Light *lights [[ buffer(BufferIndexLights) ]],
                             texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]],
                             texture2d<float, access::read> prevTex [[ texture(TextureIndexAccumulation) ]],
                             texture2d<float, access::write> dstTex [[ texture(TextureIndexPreviousAccumulation) ]],
                             texture2d<float, access::write> depthTex [[ texture(TextureIndexDepth) ]],
                             texture2d<float, access::write> motionTex [[ texture(TextureIndexMotion) ]])
{
    // The sample aligns the thread count to the threadgroup size. which means the thread count
    // may be different than the bounds of the texture. Test to make sure this thread
    // is referencing a pixel within the bounds of the texture.
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        float2 pixel = (float2)tid;
        
        // Apply a random offset to the random number index to decorrelate pixels.
        unsigned int offset = randomTexture.read(tid).x;
        
        float3 totalColor = float3(0.0f);
        
        // For temporal scaler: track first hit for depth and motion
        float primaryDepth = 0.0f;
        float2 motionVector = float2(0.0f);
        
        // Multiple samples per pixel
        for (int sampleIndex = 0; sampleIndex < uniforms.samplesPerPixel; sampleIndex++) {
            int frameOffset = uniforms.frameIndex * uniforms.samplesPerPixel + sampleIndex;
            
            // Add a random offset to the pixel coordinates for antialiasing.
            float2 r = float2(halton(offset + frameOffset, 0),
                              halton(offset + frameOffset, 1));
        pixel += r;
        
        // Map pixel coordinates to -1..1.
        float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
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
                
        for (int bounce = 0; bounce < uniforms.maxBounces; bounce++) {
            // Get the closest intersection, not the first intersection. This is the default, but
            // the sample adjusts this property below when it casts shadow rays.
            i.accept_any_intersection(false);
            
            // Check for intersection between the ray and the acceleration structure.
            //intersection = i.intersect(ray, accelerationStructure, bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY);
            intersection = i.intersect(ray, accelerationStructure);
            
            // Store depth and motion for first bounce (primary ray)
            if (bounce == 0 && sampleIndex == 0 && intersection.type != intersection_type::none) {
                // Depth from camera
                primaryDepth = intersection.distance;
                
                // Compute motion vector (current position - previous frame position)
                float3 worldPos = ray.origin + ray.direction * intersection.distance;
                
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
                float3 prevViewPos = worldPos - prevCamera.position;
                float2 prevScreenPos;
                prevScreenPos.x = dot(prevViewPos, prevCamera.right);
                prevScreenPos.y = dot(prevViewPos, prevCamera.up);
                float prevDepth = dot(prevViewPos, prevCamera.forward);
                prevScreenPos /= max(prevDepth, 0.001f);
                
                // Motion vector in screen space
                motionVector = screenPos - prevScreenPos;
            }
            
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
            
            float3 objectSpaceSurfaceNormal = interpolateVertexAttribute(resource.normals, intersection, resource.indices);
            float3 worldSpaceSurfaceNormal = (objectToWorldSpaceTransform * float4(objectSpaceSurfaceNormal, 0)).xyz;
            worldSpaceSurfaceNormal = normalize(worldSpaceSurfaceNormal);
            float3 surfaceColor = resource.material->baseColor;
            
            // Choose a random light source to sample.
            float lightSample = halton(offset + frameOffset, 2 + bounce * 5 + 0);
            int lightIndex = min((int)(lightSample * uniforms.lightCount), uniforms.lightCount - 1);
            
            device Light &light = lights[lightIndex];
            
            float3 worldSpaceLightDirection;
            float lightDistance;
            float3 lightColor;
            
            if (light.type == LightTypeAreaLight) {
                
                // Choose a random point to sample on the light source.
                r = float2(halton(offset + frameOffset, 2 + bounce * 5 + 1),
                                  halton(offset + frameOffset, 2 + bounce * 5 + 2));

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
            
            // Scale the light color by the cosine of the angle between the light direction and
            // surface normal.
            lightColor *= saturate(dot(worldSpaceSurfaceNormal, worldSpaceLightDirection));

            // Scale the light color by the number of lights to compensate for the fact that
            // the sample only samples one light source at random.
            lightColor *= uniforms.lightCount;

            // Scale the ray color by the color of the surface. This simulates light being absorbed into
            // the surface.
            color *= surfaceColor;
            
            // Early exit if color contribution becomes negligible
            if (length(color) < 0.001) {
                break;
            }
            
            if (length(lightColor) > 0.0001) {
            
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
                    accumulatedColor += lightColor * color;
                }
            }

            // Next choose a random direction to continue the path of the ray. This will
            // cause light to bounce between surfaces. The sample could apply a fair bit of math
            // to compute the fraction of light reflected by the current intersection point to the
            // previous point from the next point. However, by choosing a random direction with
            // probability proportional to the cosine (dot product) of the angle between the
            // sample direction and surface normal, the math entirely cancels out except for
            // multiplying by the surface color. This sampling strategy also reduces the amount
            // of noise in the output image.
            r = float2(halton(offset + frameOffset, 2 + bounce * 5 + 3),
                       halton(offset + frameOffset, 2 + bounce * 5 + 4));

            float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(r);
            worldSpaceSampleDirection = alignHemisphereWithNormal(worldSpaceSampleDirection, worldSpaceSurfaceNormal);

            ray.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
            ray.direction = worldSpaceSampleDirection;
        }
        
        totalColor += accumulatedColor;
    } // End of samples loop
    
    // Average all samples
    totalColor /= uniforms.samplesPerPixel;
                    
    // Average this frame's sample with all of the previous frames.
    if (uniforms.frameIndex > 0) {
        float3 prevColor = prevTex.read(tid).xyz;
        prevColor *= uniforms.frameIndex;

        totalColor += prevColor;
        totalColor /= (uniforms.frameIndex + 1);
    }

    dstTex.write(float4(totalColor, 1.0f), tid);
    
    // Write depth and motion for temporal scaler (if textures are bound)
    depthTex.write(primaryDepth, tid);
    motionTex.write(float4(motionVector, 0.0f, 0.0f), tid);
    }
}


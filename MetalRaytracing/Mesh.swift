//
//  Mesh.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

struct MeshSkinningInfo {
    let jointPaths: [String]
    let jointToSkeletonIndex: [Int]
    let geometryBindTransform: matrix_float4x4
    let geometryBindTransformInverse: matrix_float4x4
}

struct Mesh {
    let mtkMesh: MTKMesh
    var transform: matrix_float4x4
    let submeshes: [Submesh]
    let hasSkinning: Bool
    let skinning: MeshSkinningInfo?
    let name: String
    
    var vertexBuffer: MTLBuffer { mtkMesh.vertexBuffers[0].buffer }
    var normalBuffer: MTLBuffer { mtkMesh.vertexBuffers[1].buffer }
    
    // Optional skinning buffers
    var jointIndexBuffer: MTLBuffer? {
         return mtkMesh.vertexBuffers.count > 2 ? mtkMesh.vertexBuffers[2].buffer : nil
    }
    var jointWeightBuffer: MTLBuffer? {
         return mtkMesh.vertexBuffers.count > 3 ? mtkMesh.vertexBuffers[3].buffer : nil
    }
    
    // Optional texture coordinate buffer
    var uvBuffer: MTLBuffer? {
         return mtkMesh.vertexBuffers.count > 4 ? mtkMesh.vertexBuffers[4].buffer : nil
    }
    
    init(modelName: String, mdlMesh: MDLMesh, mtkMesh: MTKMesh, transform: matrix_float4x4, hasSkinning: Bool, skinning: MeshSkinningInfo? = nil, on device: MTLDevice) {
        self.mtkMesh = mtkMesh
        self.transform = transform
        self.skinning = skinning
        self.hasSkinning = skinning != nil ? true : hasSkinning
        self.name = mdlMesh.name
        
        var submeshes: [Submesh] = []
        let positionBuffer = mtkMesh.vertexBuffers[0].buffer
        let normalBuffer = mtkMesh.vertexBuffers[1].buffer
        let uvBuffer = mtkMesh.vertexBuffers.count > 4 ? mtkMesh.vertexBuffers[4].buffer : nil
        
        for mesh in zip(mdlMesh.submeshes!, mtkMesh.submeshes) {
            submeshes.append(Submesh(modelName: modelName, mdlSubmesh: mesh.0 as! MDLSubmesh, mtkSubmesh: mesh.1, positionBuffer: positionBuffer, normalBuffer: normalBuffer, uvBuffer: uvBuffer, mask: GEOMETRY_MASK_TRIANGLE, on: device))
        }

        self.submeshes = submeshes
    }
    
    // Backward compatibility init (if needed, or just remove)
    init(modelName: String, mdlMesh: MDLMesh, mtkMesh: MTKMesh, position: SIMD3<Float>, rotation: SIMD3<Float>, scale: Float, on device: MTLDevice) {
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        let scaleMatrix = matrix_float4x4.scale(scale)
        let translationMatrix = matrix_float4x4.translate(position)
        let transform = translationMatrix * rotationMatrix * scaleMatrix
        
        self.init(modelName: modelName, mdlMesh: mdlMesh, mtkMesh: mtkMesh, transform: transform, hasSkinning: false, skinning: nil, on: device)
    }
    
    var resources: [MTLResource] {
        submeshes.flatMap { $0.resources }
    }
    
    var geometryDescriptors: [MTLAccelerationStructureTriangleGeometryDescriptor] {
        submeshes.map { submesh in
            let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
            descriptor.vertexBuffer = self.vertexBuffer
            descriptor.indexBuffer = submesh.mtkSubmesh.indexBuffer.buffer
            descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            descriptor.triangleCount = submesh.mtkSubmesh.indexCount / 3
            return descriptor
        }
    }
    var geometryDescriptorsMTL4: [MTL4AccelerationStructureTriangleGeometryDescriptor] {
        return submeshes.map { submesh in
            let descriptor = MTL4AccelerationStructureTriangleGeometryDescriptor()
            // Vertex buffer (position) with offset
            let vb = mtkMesh.vertexBuffers[0]
            let vertexAddress = vb.buffer.gpuAddress + UInt64(vb.offset)
            descriptor.vertexBuffer = MTL4BufferRange(bufferAddress: vertexAddress, length: UInt64(vb.length))
            descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            descriptor.vertexFormat = .float3
            // Index buffer with offset
            let ib = submesh.mtkSubmesh.indexBuffer
            let indexAddress = ib.buffer.gpuAddress + UInt64(ib.offset)
            descriptor.indexBuffer = MTL4BufferRange(bufferAddress: indexAddress, length: UInt64(ib.length))
            descriptor.indexType = submesh.mtkSubmesh.indexType
            descriptor.triangleCount = submesh.mtkSubmesh.indexCount / 3
            return descriptor
        }
    }
}

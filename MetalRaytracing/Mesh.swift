//
//  Mesh.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

struct Mesh {
    let mtkMesh: MTKMesh
    let transform: matrix_float4x4
    let submeshes: [Submesh]
    
    var vertexBuffer: MTLBuffer { mtkMesh.vertexBuffers[0].buffer }
    var normalBuffer: MTLBuffer { mtkMesh.vertexBuffers[1].buffer }
    
    init(modelName: String, mdlMesh: MDLMesh, mtkMesh: MTKMesh, position: SIMD3<Float>, rotation: SIMD3<Float>, scale: Float, on device: MTLDevice) {
        self.mtkMesh = mtkMesh
        
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        let scaleMatrix = matrix_float4x4.scale(scale)
        let translationMatrix = matrix_float4x4.translate(position)
        self.transform = translationMatrix * rotationMatrix * scaleMatrix
        
        var submeshes: [Submesh] = []
        let normalBuffer = mtkMesh.vertexBuffers[1].buffer
        for mesh in zip(mdlMesh.submeshes!, mtkMesh.submeshes) {
            submeshes.append(Submesh(modelName: modelName, mdlSubmesh: mesh.0 as! MDLSubmesh, mtkSubmesh: mesh.1, normalBuffer: normalBuffer, mask: GEOMETRY_MASK_TRIANGLE, on: device))
        }

        self.submeshes = submeshes
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
    var geometryDescriptorsMTL4: [MTL4AccelerationStructureGeometryDescriptor] {
        submeshes.map { submesh in
            let descriptor = MTL4AccelerationStructureTriangleGeometryDescriptor()
            let vertexBuffer = mtkMesh.vertexBuffers[0]
            let vertexAddress = vertexBuffer.buffer.gpuAddress + UInt64(vertexBuffer.offset)
            descriptor.vertexBuffer = MTL4BufferRange(bufferAddress: vertexAddress, length: UInt64(vertexBuffer.length))
            descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            descriptor.vertexFormat = .float3

            let indexBuffer = submesh.mtkSubmesh.indexBuffer
            let indexAddress = indexBuffer.buffer.gpuAddress + UInt64(indexBuffer.offset)
            descriptor.indexBuffer = MTL4BufferRange(bufferAddress: indexAddress, length: UInt64(indexBuffer.length))
            descriptor.indexType = submesh.mtkSubmesh.indexType
            descriptor.triangleCount = submesh.mtkSubmesh.indexCount / 3
            return descriptor
        }
    }
}

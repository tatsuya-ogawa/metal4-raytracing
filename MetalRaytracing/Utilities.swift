//
//  Utilities.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

class CommonStorageMode{
#if os(macOS)
static let options:MTLResourceOptions = .storageModeManaged
#else
static let options:MTLResourceOptions = .storageModeShared
#endif
}

extension MTLDevice {
    func makeArgumentEncoder(for resources: [MTLResource]) -> MTLArgumentEncoder? {
        let argumentDescriptors = resources.enumerated().map { i, resource -> MTLArgumentDescriptor in
            let argumentDescriptor = MTLArgumentDescriptor()
            argumentDescriptor.index = i
            argumentDescriptor.access = .readOnly
            if resource is MTLBuffer {
                argumentDescriptor.dataType = .pointer
            } else if let texture = resource as? MTLTexture {
                argumentDescriptor.textureType = texture.textureType
                argumentDescriptor.dataType = .texture
            }
            return argumentDescriptor
        }
        return self.makeArgumentEncoder(arguments: argumentDescriptors)
    }
}

extension MTLCommandQueue {
    func buildCompactedAccelerationStructures(for descriptors: [MTLAccelerationStructureDescriptor]) -> [MTLAccelerationStructure] {
        guard
            !descriptors.isEmpty,
            let commandBuffer = self.makeCommandBuffer(),
            let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else { return [] }
        commandBuffer.label = "CommandBuffer BuildAccelerationStructures"
        encoder.label = "CommandEncoder BuildAccelerationStructures"
        
        let descriptorsAndSizes = descriptors.map { descriptor -> (MTLAccelerationStructureDescriptor, MTLAccelerationStructureSizes) in
            let sizes: MTLAccelerationStructureSizes = self.device.accelerationStructureSizes(descriptor: descriptor)
            return (descriptor, sizes)
        }
        
        let scratchBufferSize = descriptorsAndSizes.map(\.1.buildScratchBufferSize).max()!
        guard
            let scratchBuffer = self.device.makeBuffer(length: scratchBufferSize, options: .storageModePrivate),
            let compactedSizesBuffer = self.device.makeBuffer(length: MemoryLayout<UInt32>.stride * descriptors.count, options: CommonStorageMode.options)
        else { return [] }
        scratchBuffer.label = "Scratch Buffer"
        compactedSizesBuffer.label = "Compacted Sizes Buffer"
        
        let accelerationStructures = descriptorsAndSizes.enumerated().map { index, descriptorAndSizes -> MTLAccelerationStructure in
            let (descriptor, sizes) = descriptorAndSizes
            let accelerationStructure = self.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
            accelerationStructure.label = "AccelerationStructure \(index)"
            encoder.build(accelerationStructure: accelerationStructure, descriptor: descriptor, scratchBuffer: scratchBuffer, scratchBufferOffset: 0)
            encoder.writeCompactedSize(accelerationStructure: accelerationStructure, buffer: compactedSizesBuffer, offset: MemoryLayout<UInt32>.stride * index)
            return accelerationStructure
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        
        guard
            let commandBuffer = self.makeCommandBuffer(),
            let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else { return [] }
        commandBuffer.label = "CommandBuffer BuildCompactedAccelerationStructures"
        encoder.label = "CommandEncoder BuildCompactedAccelerationStructures"
        
        let compactedSizes = compactedSizesBuffer.contents().bindMemory(to: UInt32.self, capacity: descriptors.count)
        
        let compactedAccelerationStructures = accelerationStructures.enumerated().map { index, accelerationStructure -> MTLAccelerationStructure in
            let compactedAccelerationStructure = self.device.makeAccelerationStructure(size: Int(compactedSizes.advanced(by: index).pointee))!
            compactedAccelerationStructure.label = "CompactedAccelerationStructure \(index)"
            encoder.copyAndCompact(sourceAccelerationStructure: accelerationStructure, destinationAccelerationStructure: compactedAccelerationStructure)
            return compactedAccelerationStructure
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return compactedAccelerationStructures
    }
    
    func buildCompactedAccelerationStructure(with descriptor: MTLAccelerationStructureDescriptor) -> MTLAccelerationStructure? {
        return buildCompactedAccelerationStructures(for: [descriptor]).first
    }
}

extension MTL4CommandQueue {
    func buildCompactedAccelerationStructures<T: MTL4AccelerationStructureDescriptor>(for descriptors: [T], residencySet: MTLResidencySet? = nil) -> [MTLAccelerationStructure] {
        guard !descriptors.isEmpty else { return [] }

        let descriptorsAndSizes = descriptors.map { descriptor -> (T, MTLAccelerationStructureSizes) in
            let sizes = device.accelerationStructureSizes(descriptor: descriptor)
            return (descriptor, sizes)
        }

        let scratchBufferSize = descriptorsAndSizes.map(\.1.buildScratchBufferSize).max()!
        guard
            let scratchBuffer = device.makeBuffer(length: scratchBufferSize, options: .storageModePrivate),
            let compactedSizesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * descriptors.count, options: .storageModeShared)
        else { return [] }
        scratchBuffer.label = "Scratch Buffer"
        compactedSizesBuffer.label = "Compacted Sizes Buffer"

        func bufferRange(_ buffer: MTLBuffer, offset: Int = 0, length: Int? = nil) -> MTL4BufferRange {
            let rangeLength = length ?? (buffer.length - offset)
            return MTL4BufferRange(bufferAddress: buffer.gpuAddress + UInt64(offset), length: UInt64(rangeLength))
        }

        func commitAndWait(_ commandBuffer: MTL4CommandBuffer) {
            let event = device.makeSharedEvent()!
            event.signaledValue = 0
            commit([commandBuffer])
            signalEvent(event, value: 1)
            while !event.wait(untilSignaledValue: 1, timeoutMS: 1000) {}
        }

        guard let buildAllocator = device.makeCommandAllocator(),
              let buildCommandBuffer = device.makeCommandBuffer()
        else { return [] }
        buildCommandBuffer.label = "CommandBuffer BuildAccelerationStructures"
        buildCommandBuffer.beginCommandBuffer(allocator: buildAllocator)
        guard let buildEncoder = buildCommandBuffer.makeComputeCommandEncoder() else { return [] }
        buildEncoder.label = "CommandEncoder BuildAccelerationStructures"

        let scratchRange = bufferRange(scratchBuffer)

        var accelerationStructures: [MTLAccelerationStructure] = []
        accelerationStructures.reserveCapacity(descriptorsAndSizes.count)

        // Create a local residency set for the scratch buffer and new acceleration structures
        let localResidencySetDescriptor = MTLResidencySetDescriptor()
        localResidencySetDescriptor.initialCapacity = descriptors.count + 2
        let localResidencySet = try! device.makeResidencySet(descriptor: localResidencySetDescriptor)
//        localResidencySet.label = "Build Acceleration Structure Local Residency Set"
        
        var allocations: [any MTLAllocation] = [scratchBuffer, compactedSizesBuffer]

        for (index, descriptorAndSizes) in descriptorsAndSizes.enumerated() {
            let (descriptor, sizes) = descriptorAndSizes
            let accelerationStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
            accelerationStructures.append(accelerationStructure)
            allocations.append(accelerationStructure)
            accelerationStructure.label = "AccelerationStructure \(index)"
            buildEncoder.build(destinationAccelerationStructure: accelerationStructure, descriptor: descriptor, scratchBuffer: scratchRange)
            let sizeOffset = MemoryLayout<UInt64>.stride * index
            let sizeRange = bufferRange(compactedSizesBuffer, offset: sizeOffset, length: MemoryLayout<UInt64>.stride)
            buildEncoder.writeCompactedSize(sourceAccelerationStructure: accelerationStructure, destinationBuffer: sizeRange)
        }
        
        localResidencySet.addAllocations(allocations)
        localResidencySet.commit()
        
        if let residencySet = residencySet {
            buildCommandBuffer.useResidencySet(residencySet)
        }
        buildCommandBuffer.useResidencySet(localResidencySet)

        buildEncoder.endEncoding()
        buildCommandBuffer.endCommandBuffer()
        commitAndWait(buildCommandBuffer)

        let compactedSizes = compactedSizesBuffer.contents().bindMemory(to: UInt64.self, capacity: descriptors.count)

        guard let compactAllocator = device.makeCommandAllocator(),
              let compactCommandBuffer = device.makeCommandBuffer()
        else { return [] }
        compactCommandBuffer.label = "CommandBuffer BuildCompactedAccelerationStructures"
        compactCommandBuffer.beginCommandBuffer(allocator: compactAllocator)
        guard let compactEncoder = compactCommandBuffer.makeComputeCommandEncoder() else { return [] }
        compactEncoder.label = "CommandEncoder BuildCompactedAccelerationStructures"

        // Ensure residency for compaction (local: sizes buffer, source AS; new: compacted AS)
        if let residencySet = residencySet {
            compactCommandBuffer.useResidencySet(residencySet)
        }
        compactCommandBuffer.useResidencySet(localResidencySet)
        
        let compactResidencySetDescriptor = MTLResidencySetDescriptor()
        compactResidencySetDescriptor.initialCapacity = descriptors.count
        let compactResidencySet = try! device.makeResidencySet(descriptor: compactResidencySetDescriptor)
//        compactResidencySet.label = "Compacted Acceleration Structure Residency Set"
        var compactAllocations: [any MTLAllocation] = []

        let compactedAccelerationStructures = accelerationStructures.enumerated().map { index, accelerationStructure -> MTLAccelerationStructure in
            let compactedSize = Int(compactedSizes.advanced(by: index).pointee)
            let compactedAccelerationStructure = device.makeAccelerationStructure(size: compactedSize)!
            compactedAccelerationStructure.label = "CompactedAccelerationStructure \(index)"
            compactEncoder.copyAndCompact(sourceAccelerationStructure: accelerationStructure, destinationAccelerationStructure: compactedAccelerationStructure)
            compactAllocations.append(compactedAccelerationStructure)
            return compactedAccelerationStructure
        }
        
        compactResidencySet.addAllocations(compactAllocations)
        compactResidencySet.commit()
        compactCommandBuffer.useResidencySet(compactResidencySet)

        compactEncoder.endEncoding()
        compactCommandBuffer.endCommandBuffer()
        commitAndWait(compactCommandBuffer)

        return compactedAccelerationStructures
    }

    func buildCompactedAccelerationStructure<T: MTL4AccelerationStructureDescriptor>(with descriptor: T, residencySet: MTLResidencySet? = nil) -> MTLAccelerationStructure? {
        return buildCompactedAccelerationStructures(for: [descriptor], residencySet: residencySet).first
    }
}


extension MTLPackedFloat4x3 {
    static func matrix4x4_drop_last_row(_ m: matrix_float4x4) -> MTLPackedFloat4x3 {
        return MTLPackedFloat4x3.init(columns: (
            MTLPackedFloat3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
            MTLPackedFloat3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
            MTLPackedFloat3(m.columns.2.x, m.columns.2.y, m.columns.2.z),
            MTLPackedFloat3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        ))
    }
}

extension MTLPackedFloat3 {
    init(_ x: Float, _ y: Float, _ z: Float) {
        var p = MTLPackedFloat3()
        p.x = x
        p.y = y
        p.z = z
        self = p
    }
}

extension matrix_float4x4 {
    static func translate(_ t: vector_float3) -> matrix_float4x4 {
        return .init(columns: (
            [  1,   0,   0, 0],
            [  0,   1,   0, 0],
            [  0,   0,   1, 0],
            [t.x, t.y, t.z, 1]
        ))
    }
    
    static func rotate(radians: Float, axis: vector_float3) -> matrix_float4x4 {
        let axis = normalize(axis)
        let ct = cosf(radians)
        let st = sinf(radians)
        let ci = 1 - ct
        let x = axis.x, y = axis.y, z = axis.z
        
        return .init(columns: (
            [ ct + x * x * ci,     y * x * ci + z * st, z * x * ci - y * st, 0],
            [ x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0],
            [ x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0],
            [                   0,                   0,                   0, 1]
        ))
    }
    
    static func rotateX(_ radians: Float) -> matrix_float4x4 {
        rotate(radians: radians, axis: [1,0,0])
    }
    
    static func rotateY(_ radians: Float) -> matrix_float4x4 {
        rotate(radians: radians, axis: [0,1,0])
    }
    
    static func rotateZ(_ radians: Float) -> matrix_float4x4 {
        rotate(radians: radians, axis: [0,0,1])
    }
    
    static func rotate(_ r: vector_float3) -> matrix_float4x4 {
        rotateX(r.x) * rotateY(r.y) * rotateZ(r.z)
    }
    
    static func scale(_ s: vector_float3) -> matrix_float4x4 {
        return .init(columns: (
            [s.x,   0,   0, 0],
            [0,   s.y,   0, 0],
            [0,     0, s.z, 0],
            [0,     0,   0, 1]
        ))
    }
    
    static func scale(_ s: Float) -> matrix_float4x4 {
        scale([s, s, s])
    }
}

extension SIMD4 {
    var xyz: SIMD3<Scalar> {
        return [self.x, self.y, self.z]
    }
}

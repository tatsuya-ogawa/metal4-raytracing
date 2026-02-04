//
//  SkinningPass.swift
//  MetalRaytracing
//
//  Created by Tatsuya Ogawa on 31/01/2026.
//

import Metal
import simd
import QuartzCore

class SkinningPass {
    let device: MTLDevice
    
    // Skinning
    var skinningTable: MTL4ArgumentTable!
    var skinningPipelineState: MTLComputePipelineState!
    // Persistent per-skinned-mesh joint matrices buffers (retain across frames)
    var jointMatrixBuffers: [MTLBuffer] = []
    // Persistent per-skinned-mesh uniform buffers (vertexCount)
    var skinningUniformBuffers: [MTLBuffer] = []
    
    // Per-mesh skinned buffers.
    // Key: Mesh identifier (or just simple parallel arrays if efficient)
    var skinnedVertexBuffers: [MTLBuffer] = []
    var skinnedNormalBuffers: [MTLBuffer] = []
    var skinnedPrevVertexBuffers: [MTLBuffer] = []
    // To map which mesh uses which buffer
    var skinnedMeshIndices: [Int] = []
    
    // Throttle skinning/BLAS updates to a fixed interval.
    private let skinningDeltaTime: Double = 1.0 / 60.0
    private var lastSkinningUpdateTime: Double = 0
    
    init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        
        // Skinning Table
        let skinningTableDesc = MTL4ArgumentTableDescriptor()
        skinningTableDesc.maxBufferBindCount = 20 // Covers indices 10-16
        self.skinningTable = try! device.makeArgumentTable(descriptor: skinningTableDesc)
        
        // Skinning Pipeline
        do {
            guard let skinningFunction = library.makeFunction(name: "skinningKernel") else {
                fatalError("Could not find skinningKernel")
            }
            skinningPipelineState = try device.makeComputePipelineState(function: skinningFunction)
        } catch {
            fatalError("Could not create skinning pipeline: \(error)")
        }
    }
    
    func createBuffers(scene: Scene) {
        // Create skinned buffers (and persistent per-mesh uniform buffers)
        skinnedVertexBuffers.removeAll(keepingCapacity: true)
        skinnedNormalBuffers.removeAll(keepingCapacity: true)
        skinnedPrevVertexBuffers.removeAll(keepingCapacity: true)
        skinnedMeshIndices.removeAll(keepingCapacity: true)
        skinningUniformBuffers.removeAll(keepingCapacity: true)
        jointMatrixBuffers.removeAll(keepingCapacity: true)
        
        for (i, model) in scene.models.enumerated() {
            for mesh in model.meshes {
                if mesh.hasSkinning {
                    let jointCount: Int
                    if let skinning = mesh.skinning, !skinning.jointPaths.isEmpty {
                        jointCount = skinning.jointPaths.count
                    } else if let skeleton = model.skeleton {
                        jointCount = skeleton.restTransforms.count
                    } else {
                        jointCount = 0
                    }
                    let safeJointCount = max(1, jointCount)
                    let jointLength = safeJointCount * MemoryLayout<matrix_float4x4>.stride
                    let jointBuffer = device.makeBuffer(length: jointLength, options: CommonStorageMode.options)!
                    jointBuffer.label = "Joint Matrices - model\(i)"
                    let jointPointer = jointBuffer.contents().bindMemory(to: matrix_float4x4.self, capacity: safeJointCount)
                    for j in 0..<safeJointCount {
                        jointPointer[j] = matrix_identity_float4x4
                    }
#if os(macOS)
                    jointBuffer.didModifyRange(0..<jointLength)
#endif
                    
                    // Create destination buffers
                    let vertexCount = mesh.mtkMesh.vertexCount
                    let posSize = vertexCount * MemoryLayout<SIMD3<Float>>.stride
                    
                    let posBuffer = device.makeBuffer(length: posSize, options: .storageModePrivate)!
                    posBuffer.label = "Skinned Position - model\(i)"
                    
                    let prevPosBuffer = device.makeBuffer(length: posSize, options: .storageModePrivate)!
                    prevPosBuffer.label = "Skinned Prev Position - model\(i)"
                    
                    let nrmBuffer = device.makeBuffer(length: posSize, options: .storageModePrivate)!
                    nrmBuffer.label = "Skinned Normal - model\(i)"
                    
                    var vertexCountU32 = UInt32(vertexCount)
                    let uniformBuffer = device.makeBuffer(bytes: &vertexCountU32,
                                                          length: MemoryLayout<UInt32>.stride,
                                                          options: CommonStorageMode.options)!
                    uniformBuffer.label = "Skinning Uniforms - model\(i)"
                    
#if os(macOS)
                    uniformBuffer.didModifyRange(0..<uniformBuffer.length)
#endif
                    
                    skinnedVertexBuffers.append(posBuffer)
                    skinnedPrevVertexBuffers.append(prevPosBuffer)
                    skinnedNormalBuffers.append(nrmBuffer)
                    skinningUniformBuffers.append(uniformBuffer)
                    jointMatrixBuffers.append(jointBuffer)
                    skinnedMeshIndices.append(skinnedMeshIndices.count) // Just tracking existence for now
                } else {
                    // Keep placeholders to align indices or handle differently
                    // Simplest is to only store for skinned meshes and track the mapping
                }
            }
        }
    }
    
    @discardableResult
    func updateSkinningJointMatrices(model: Model, mesh: Mesh, destinationBuffer: MTLBuffer) -> Int {
        let stride = MemoryLayout<matrix_float4x4>.stride
        let capacity = destinationBuffer.length / stride
        guard capacity > 0 else { return 0 }
        
        let skinMatrices = model.jointMatrices
        let mapping = mesh.skinning?.jointToSkeletonIndex ?? []
        let geometryBind = mesh.skinning?.geometryBindTransform ?? matrix_identity_float4x4
        let geometryBindInverse = mesh.skinning?.geometryBindTransformInverse ?? matrix_identity_float4x4
        
        let jointCount = min(capacity, mapping.isEmpty ? max(1, skinMatrices.count) : mapping.count)
        let dst = destinationBuffer.contents().bindMemory(to: matrix_float4x4.self, capacity: jointCount)
        
        if skinMatrices.isEmpty {
            for i in 0..<jointCount {
                dst[i] = matrix_identity_float4x4
            }
        } else {
            for i in 0..<jointCount {
                let skeletonIndex = mapping.isEmpty ? i : mapping[i]
                let skinMatrix: matrix_float4x4
                if skeletonIndex >= 0 && skeletonIndex < skinMatrices.count {
                    skinMatrix = skinMatrices[skeletonIndex]
                } else {
                    skinMatrix = matrix_identity_float4x4
                }
                dst[i] = simd_mul(geometryBindInverse, simd_mul(skinMatrix, geometryBind))
            }
        }
#if os(macOS)
        destinationBuffer.didModifyRange(0..<(jointCount * stride))
#endif
        return jointCount
    }
    
    /// Dispatch skinning kernel for all skinned meshes using provided encoder
    func dispatchSkinning(computeEncoder: MTL4ComputeCommandEncoder, scene: Scene) {
        computeEncoder.setComputePipelineState(skinningPipelineState)
        
        var skinnedIndex = 0
        for model in scene.models {
            for mesh in model.meshes where mesh.hasSkinning {
                guard skinnedIndex < jointMatrixBuffers.count else { continue }
                
                // Joint matrices
                let jointBuffer = jointMatrixBuffers[skinnedIndex]
                updateSkinningJointMatrices(model: model, mesh: mesh, destinationBuffer: jointBuffer)
                skinningTable.setAddress(jointBuffer.gpuAddress, index: BufferIndex.jointMatrices.rawValue)
                
                // Source buffers
                let restPositionsVB = mesh.mtkMesh.vertexBuffers[0]
                let restNormalsVB = mesh.mtkMesh.vertexBuffers[1]
                skinningTable.setAddress(restPositionsVB.buffer.gpuAddress + UInt64(restPositionsVB.offset),
                                         index: BufferIndex.restPositions.rawValue)
                skinningTable.setAddress(restNormalsVB.buffer.gpuAddress + UInt64(restNormalsVB.offset),
                                         index: BufferIndex.restNormals.rawValue)
                
                if mesh.mtkMesh.vertexBuffers.count > 2 {
                    let jointIndexVB = mesh.mtkMesh.vertexBuffers[2]
                    skinningTable.setAddress(jointIndexVB.buffer.gpuAddress + UInt64(jointIndexVB.offset),
                                             index: BufferIndex.jointIndices.rawValue)
                }
                if mesh.mtkMesh.vertexBuffers.count > 3 {
                    let jointWeightVB = mesh.mtkMesh.vertexBuffers[3]
                    skinningTable.setAddress(jointWeightVB.buffer.gpuAddress + UInt64(jointWeightVB.offset),
                                             index: BufferIndex.jointWeights.rawValue)
                }
                
                // Destination buffers
                let destPos = skinnedVertexBuffers[skinnedIndex]
                let destNrm = skinnedNormalBuffers[skinnedIndex]
                skinningTable.setAddress(destPos.gpuAddress, index: BufferIndex.skinnedPositions.rawValue)
                skinningTable.setAddress(destNrm.gpuAddress, index: BufferIndex.skinnedNormals.rawValue)
                
                // Uniforms
                skinningTable.setAddress(skinningUniformBuffers[skinnedIndex].gpuAddress, index: BufferIndex.uniforms.rawValue)
                
                // Dispatch
                computeEncoder.setArgumentTable(skinningTable)
                let vertexCount = mesh.mtkMesh.vertexCount
                let threadsPerGrid = MTLSize(width: vertexCount, height: 1, depth: 1)
                let threadsPerGroup = MTLSize(width: min(vertexCount, skinningPipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
                
                skinnedIndex += 1
            }
        }
    }

    /// Dispatch skinning kernel for all skinned meshes using legacy encoder
    func dispatchSkinningLegacy(computeEncoder: MTLComputeCommandEncoder, scene: Scene) {
        computeEncoder.setComputePipelineState(skinningPipelineState)

        var skinnedIndex = 0
        for model in scene.models {
            for mesh in model.meshes where mesh.hasSkinning {
                guard skinnedIndex < jointMatrixBuffers.count else { continue }

                // Joint matrices
                let jointBuffer = jointMatrixBuffers[skinnedIndex]
                updateSkinningJointMatrices(model: model, mesh: mesh, destinationBuffer: jointBuffer)
                computeEncoder.setBuffer(jointBuffer, offset: 0, index: BufferIndex.jointMatrices.rawValue)

                // Source buffers
                let restPositionsVB = mesh.mtkMesh.vertexBuffers[0]
                let restNormalsVB = mesh.mtkMesh.vertexBuffers[1]
                computeEncoder.setBuffer(restPositionsVB.buffer,
                                         offset: restPositionsVB.offset,
                                         index: BufferIndex.restPositions.rawValue)
                computeEncoder.setBuffer(restNormalsVB.buffer,
                                         offset: restNormalsVB.offset,
                                         index: BufferIndex.restNormals.rawValue)

                if mesh.mtkMesh.vertexBuffers.count > 2 {
                    let jointIndexVB = mesh.mtkMesh.vertexBuffers[2]
                    computeEncoder.setBuffer(jointIndexVB.buffer,
                                             offset: jointIndexVB.offset,
                                             index: BufferIndex.jointIndices.rawValue)
                } else {
                    computeEncoder.setBuffer(nil, offset: 0, index: BufferIndex.jointIndices.rawValue)
                }
                if mesh.mtkMesh.vertexBuffers.count > 3 {
                    let jointWeightVB = mesh.mtkMesh.vertexBuffers[3]
                    computeEncoder.setBuffer(jointWeightVB.buffer,
                                             offset: jointWeightVB.offset,
                                             index: BufferIndex.jointWeights.rawValue)
                } else {
                    computeEncoder.setBuffer(nil, offset: 0, index: BufferIndex.jointWeights.rawValue)
                }

                // Destination buffers
                let destPos = skinnedVertexBuffers[skinnedIndex]
                let destNrm = skinnedNormalBuffers[skinnedIndex]
                computeEncoder.setBuffer(destPos, offset: 0, index: BufferIndex.skinnedPositions.rawValue)
                computeEncoder.setBuffer(destNrm, offset: 0, index: BufferIndex.skinnedNormals.rawValue)

                // Uniforms
                computeEncoder.setBuffer(skinningUniformBuffers[skinnedIndex],
                                         offset: 0,
                                         index: BufferIndex.uniforms.rawValue)

                // Dispatch
                let vertexCount = mesh.mtkMesh.vertexCount
                let threadsPerGrid = MTLSize(width: vertexCount, height: 1, depth: 1)
                let threadsPerGroup = MTLSize(width: min(vertexCount, skinningPipelineState.maxTotalThreadsPerThreadgroup),
                                              height: 1,
                                              depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

                skinnedIndex += 1
            }
        }
    }
    
    func performSkinning(commandBuffer: MTL4CommandBuffer, scene: Scene) {
        let skinnedMeshes = scene.models.flatMap { $0.meshes }.filter { $0.hasSkinning }
        if skinnedMeshes.isEmpty { return }
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Skinning"
        dispatchSkinning(computeEncoder: computeEncoder, scene: scene)
        computeEncoder.endEncoding()
    }
    
    func updateSceneTimeAndAnimation(scene: Scene) -> Bool {
        // 1. Update Animation Time & Matrices
        let now = CACurrentMediaTime()
        if lastSkinningUpdateTime == 0 {
            // Force an update on the first call.
            lastSkinningUpdateTime = now - skinningDeltaTime
        }
        let elapsed = now - lastSkinningUpdateTime
        guard elapsed >= skinningDeltaTime || scene.isDirty else { return false }
        
        // Reset dirty flag if we proceeding
        if scene.isDirty {
            scene.isDirty = false
        }
        
        let steps = Int(elapsed / skinningDeltaTime)
        let deltaTime = skinningDeltaTime * Double(steps)
        if steps > 0 {
            lastSkinningUpdateTime += skinningDeltaTime * Double(steps)
        }
        for model in scene.models {
            model.update(deltaTime: deltaTime)
        }
        return true
    }
    
    func collectAllocations(allocations: inout [any MTLAllocation]) {
        for buffer in skinnedVertexBuffers {
            allocations.append(buffer)
        }
        for buffer in skinnedPrevVertexBuffers {
            allocations.append(buffer)
        }
        for buffer in skinnedNormalBuffers {
            allocations.append(buffer)
        }
        for buffer in skinningUniformBuffers {
            allocations.append(buffer)
        }
        for buffer in jointMatrixBuffers {
            allocations.append(buffer)
        }
    }
}

import Foundation
import Metal
import simd

private enum SkinningDebug {
    struct InputStats {
        var zeroWeightCount = 0
        var weightSumOutOfRangeCount = 0
        var outOfRangeJointIndexCount = 0
        var abnormalWeightCount = 0  // weights > 1.0 or < 0
        var maxWeight: Float = 0
        var minWeight: Float = Float.greatestFiniteMagnitude
    }

    static func cpuSkinning(vertexCount: Int,
                            restPositions: [SIMD3<Float>],
                            restNormals: [SIMD3<Float>],
                            jointIndices: [SIMD4<UInt16>],
                            jointWeights: [SIMD4<Float>],
                            jointMatrices: [matrix_float4x4]) -> (positions: [SIMD3<Float>], normals: [SIMD3<Float>], stats: InputStats) {
        let safeCount = min(vertexCount,
                            restPositions.count,
                            restNormals.count,
                            jointIndices.count,
                            jointWeights.count)
        var stats = InputStats()
        var skinnedPositions = [SIMD3<Float>](repeating: .zero, count: safeCount)
        var skinnedNormals = [SIMD3<Float>](repeating: .zero, count: safeCount)

        for i in 0..<safeCount {
            let position = restPositions[i]
            let normal = restNormals[i]
            let indices = jointIndices[i]
            let weights = jointWeights[i]
            
            // Track weight statistics
            for j in 0..<4 {
                let w = weights[j]
                if w > stats.maxWeight { stats.maxWeight = w }
                if w < stats.minWeight { stats.minWeight = w }
                if w > 1.0 || w < 0 {
                    stats.abnormalWeightCount += 1
                }
            }
            
            // Correct: use addition for sum, not multiplication
            let weightSum = weights.x + weights.y + weights.z + weights.w
            
            if weightSum == 0 {
                stats.zeroWeightCount += 1
            } else if abs(weightSum - 1.0) > 0.01 {
                stats.weightSumOutOfRangeCount += 1
            }

            var skinnedPos = SIMD4<Float>(repeating: 0)
            var skinnedNrm = SIMD3<Float>(repeating: 0)

            for j in 0..<4 {
                let weight = weights[j]
                if weight > 0 {
                    let index = Int(indices[j])
                    if index < jointMatrices.count {
                        let mat = jointMatrices[index]
                        let pos4 = SIMD4<Float>(position.x, position.y, position.z, 1.0)
                        let nrm4 = SIMD4<Float>(normal.x, normal.y, normal.z, 0.0)
                        skinnedPos += weight * (mat * pos4)
                        let transformedNormal = mat * nrm4
                        skinnedNrm += weight * SIMD3<Float>(transformedNormal.x, transformedNormal.y, transformedNormal.z)
                    } else {
                        stats.outOfRangeJointIndexCount += 1
                    }
                }
            }

            skinnedPositions[i] = SIMD3<Float>(skinnedPos.x, skinnedPos.y, skinnedPos.z)
            skinnedNormals[i] = skinnedNrm
        }

        return (skinnedPositions, skinnedNormals, stats)
    }

    static func readBuffer<T>(buffer: MTLBuffer, offset: Int, count: Int) -> [T] {
        guard buffer.storageMode != .private else { return [] }
        let stride = MemoryLayout<T>.stride
        let available = max(0, (buffer.length - offset) / stride)
        let safeCount = min(count, available)
        guard safeCount > 0 else { return [] }
        let pointer = buffer.contents().advanced(by: offset).bindMemory(to: T.self, capacity: safeCount)
        return Array(UnsafeBufferPointer(start: pointer, count: safeCount))
    }

    static func readIndexBuffer(buffer: MTLBuffer, offset: Int, count: Int, indexType: MTLIndexType) -> [UInt32] {
        guard buffer.storageMode != .private else { return [] }
        switch indexType {
        case .uint16:
            let stride = MemoryLayout<UInt16>.stride
            let available = max(0, (buffer.length - offset) / stride)
            let safeCount = min(count, available)
            guard safeCount > 0 else { return [] }
            let pointer = buffer.contents().advanced(by: offset).bindMemory(to: UInt16.self, capacity: safeCount)
            var indices: [UInt32] = []
            indices.reserveCapacity(safeCount)
            for i in 0..<safeCount {
                indices.append(UInt32(pointer[i]))
            }
            return indices
        case .uint32:
            let stride = MemoryLayout<UInt32>.stride
            let available = max(0, (buffer.length - offset) / stride)
            let safeCount = min(count, available)
            guard safeCount > 0 else { return [] }
            let pointer = buffer.contents().advanced(by: offset).bindMemory(to: UInt32.self, capacity: safeCount)
            return Array(UnsafeBufferPointer(start: pointer, count: safeCount))
        @unknown default:
            return []
        }
    }

    static func exportOBJ(modelIndex: Int,
                          meshIndex: Int,
                          mesh: Mesh,
                          positions: [SIMD3<Float>],
                          normals: [SIMD3<Float>],
                          to directoryURL: URL) {
        let safeName = mesh.name.isEmpty ? "mesh\(meshIndex)" : mesh.name
        let fileURL = directoryURL.appendingPathComponent("skinning_cpu_model\(modelIndex)_\(safeName).obj")
        var output = "# Skinning CPU export\n"

        output.reserveCapacity(max(positions.count, normals.count) * 64)

        for p in positions {
            output.append("v \(p.x) \(p.y) \(p.z)\n")
        }
        let hasNormals = normals.count == positions.count
        if hasNormals {
            for n in normals {
                output.append("vn \(n.x) \(n.y) \(n.z)\n")
            }
        }

        var wroteFaces = false
        for submesh in mesh.mtkMesh.submeshes {
            let indices = readIndexBuffer(buffer: submesh.indexBuffer.buffer,
                                          offset: submesh.indexBuffer.offset,
                                          count: submesh.indexCount,
                                          indexType: submesh.indexType)
            guard !indices.isEmpty else { continue }
            wroteFaces = true
            var i = 0
            while i + 2 < indices.count {
                let raw0 = Int(indices[i])
                let raw1 = Int(indices[i + 1])
                let raw2 = Int(indices[i + 2])
                if raw0 >= positions.count || raw1 >= positions.count || raw2 >= positions.count {
                    i += 3
                    continue
                }
                let i0 = raw0 + 1
                let i1 = raw1 + 1
                let i2 = raw2 + 1
                if hasNormals {
                    output.append("f \(i0)//\(i0) \(i1)//\(i1) \(i2)//\(i2)\n")
                } else {
                    output.append("f \(i0) \(i1) \(i2)\n")
                }
                i += 3
            }
        }

        if !wroteFaces {
            output.append("# No index data available for faces\n")
        }

        do {
            try output.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Exported CPU skinning OBJ: \(fileURL.path)")
        } catch {
            print("Failed to export OBJ for model \(modelIndex) mesh \(meshIndex): \(error)")
        }
    }

    static func appendToMergedOBJ(output: inout String,
                                  modelIndex: Int,
                                  meshIndex: Int,
                                  mesh: Mesh,
                                  positions: [SIMD3<Float>],
                                  normals: [SIMD3<Float>],
                                  vertexOffset: inout Int,
                                  normalOffset: inout Int,
                                  wroteFaces: inout Bool) {
        guard !positions.isEmpty else { return }

        let vertexBase = vertexOffset
        let normalBase = normalOffset
        let hasNormals = normals.count == positions.count && !normals.isEmpty
        let safeName = mesh.name.isEmpty ? "mesh\(meshIndex)" : mesh.name

        output.append("o model\(modelIndex)_\(safeName)\n")

        for p in positions {
            output.append("v \(p.x) \(p.y) \(p.z)\n")
        }
        if hasNormals {
            for n in normals {
                output.append("vn \(n.x) \(n.y) \(n.z)\n")
            }
        }

        for submesh in mesh.mtkMesh.submeshes {
            let indices = readIndexBuffer(buffer: submesh.indexBuffer.buffer,
                                          offset: submesh.indexBuffer.offset,
                                          count: submesh.indexCount,
                                          indexType: submesh.indexType)
            guard !indices.isEmpty else { continue }
            var i = 0
            while i + 2 < indices.count {
                let raw0 = Int(indices[i])
                let raw1 = Int(indices[i + 1])
                let raw2 = Int(indices[i + 2])
                if raw0 >= positions.count || raw1 >= positions.count || raw2 >= positions.count {
                    i += 3
                    continue
                }
                let v0 = raw0 + 1 + vertexBase
                let v1 = raw1 + 1 + vertexBase
                let v2 = raw2 + 1 + vertexBase
                if hasNormals {
                    let n0 = raw0 + 1 + normalBase
                    let n1 = raw1 + 1 + normalBase
                    let n2 = raw2 + 1 + normalBase
                    output.append("f \(v0)//\(n0) \(v1)//\(n1) \(v2)//\(n2)\n")
                } else {
                    output.append("f \(v0) \(v1) \(v2)\n")
                }
                wroteFaces = true
                i += 3
            }
        }

        vertexOffset += positions.count
        if hasNormals {
            normalOffset += normals.count
        }
    }

    static func exportMergedOBJ(output: String, wroteFaces: Bool, to directoryURL: URL) {
        let fileURL = directoryURL.appendingPathComponent("skinning_cpu_merged.obj")
        var finalOutput = output
        if !wroteFaces {
            finalOutput.append("# No index data available for faces\n")
        }
        do {
            try finalOutput.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Exported CPU skinning merged OBJ: \(fileURL.path)")
        } catch {
            print("Failed to export merged OBJ: \(error)")
        }
    }

    static func countZeroVectors(_ values: [SIMD3<Float>], epsilon: Float) -> Int {
        var count = 0
        for v in values {
            if abs(v.x) <= epsilon && abs(v.y) <= epsilon && abs(v.z) <= epsilon {
                count += 1
            }
        }
        return count
    }

    static func countInvalidVectors(_ values: [SIMD3<Float>]) -> Int {
        var count = 0
        for v in values {
            if !v.x.isFinite || !v.y.isFinite || !v.z.isFinite {
                count += 1
            }
        }
        return count
    }

    static func countZeroMatrices(_ matrices: [matrix_float4x4]) -> Int {
        var count = 0
        for m in matrices {
            if m.columns.0 == .zero && m.columns.1 == .zero && m.columns.2 == .zero && m.columns.3 == .zero {
                count += 1
            }
        }
        return count
    }

    static func checkGeometricConsistency(mesh: Mesh, positions: [SIMD3<Float>], normals: [SIMD3<Float>], logLimit: Int = 10) {
        print("--- Geometric Consistency Check: \(mesh.name) ---")
        var inconsistencies = 0
        var zeroNormals = 0
        var flippedNormals = 0
        
        for submesh in mesh.mtkMesh.submeshes {
            let indices = readIndexBuffer(buffer: submesh.indexBuffer.buffer,
                                          offset: submesh.indexBuffer.offset,
                                          count: submesh.indexCount,
                                          indexType: submesh.indexType)
            guard !indices.isEmpty else { continue }
            
            var i = 0
            while i + 2 < indices.count {
                let idx0 = Int(indices[i])
                let idx1 = Int(indices[i+1])
                let idx2 = Int(indices[i+2])
                
                if idx0 >= positions.count || idx1 >= positions.count || idx2 >= positions.count {
                    i += 3; continue
                }
                
                let v0 = positions[idx0]
                let v1 = positions[idx1]
                let v2 = positions[idx2]
                
                let edge1 = v1 - v0
                let edge2 = v2 - v0
                let crossProd = simd_cross(edge1, edge2)
                let area = simd_length(crossProd)
                
                if area < 1e-6 {
                    // Degenerate triangle, skip
                    i += 3
                    continue
                }
                
                let faceNormal = normalize(crossProd)
                
                // key: index in positions/normals
                for idx in [idx0, idx1, idx2] {
                    let n = normals[idx]
                    let len = simd_length(n)
                    
                    if len < 0.001 {
                        if zeroNormals < logLimit {
                            print("Zero Normal at vertex \(idx): \(n)")
                        }
                        zeroNormals += 1
                    } else {
                        // Check alignment with face normal
                        let dot = simd_dot(normalize(n), faceNormal)
                        if dot < 0.0 {
                            if flippedNormals < logLimit {
                                print("Flipped Normal at vertex \(idx) (dot: \(dot)): Face=\(faceNormal), Vertex=\(n)")
                            }
                            flippedNormals += 1
                        }
                    }
                }
                i += 3
            }
        }
        
        if zeroNormals > 0 {
            print("WARNING: Found \(zeroNormals) zero normals")
        }
        if flippedNormals > 0 {
            print("WARNING: Found \(flippedNormals) normals opposing face normal")
        }
        if zeroNormals == 0 && flippedNormals == 0 {
            print("Geometric check passed: Normals seem consistent with geometry.")
        }
    }
}

extension Renderer {
    func runSkinningDebugTests(epsilon: Float = 0.001, logLimit: Int = 5, exportDirectory: URL? = nil) {
        print("=== Skinning Debug Tests ===")

        let skinnedMeshes = scene.models.flatMap { $0.meshes }.filter { $0.hasSkinning }
        guard !skinnedMeshes.isEmpty else {
            print("No skinned meshes found.")
            return
        }

        for model in scene.models {
            model.update(deltaTime: 0)
        }

        struct DebugMeshRecord {
            let modelIndex: Int
            let meshIndex: Int
            let mesh: Mesh
            let positionBuffer: MTLBuffer
            let normalBuffer: MTLBuffer
            let jointMatrixBuffer: MTLBuffer
        }

        var debugMeshes: [DebugMeshRecord] = []
        debugMeshes.reserveCapacity(skinnedMeshes.count)

        var debugSkinnedIndex = 0
        for (modelIndex, model) in scene.models.enumerated() {
            for (meshIndex, mesh) in model.meshes.enumerated() where mesh.hasSkinning {
                guard debugSkinnedIndex < jointMatrixBuffers.count else {
                    print("Missing joint matrix buffer for model \(modelIndex) mesh \(meshIndex)")
                    debugSkinnedIndex += 1
                    continue
                }
                let vertexCount = mesh.mtkMesh.vertexCount
                let byteCount = vertexCount * MemoryLayout<SIMD3<Float>>.stride
                guard let posBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
                      let nrmBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                    print("Failed to allocate debug buffers for model \(modelIndex) mesh \(meshIndex)")
                    debugSkinnedIndex += 1
                    continue
                }
                posBuffer.label = "Skinning Debug Positions \(modelIndex):\(mesh.name)"
                nrmBuffer.label = "Skinning Debug Normals \(modelIndex):\(mesh.name)"
                let jointMatrixBuffer = jointMatrixBuffers[debugSkinnedIndex]
                debugMeshes.append(DebugMeshRecord(modelIndex: modelIndex,
                                                   meshIndex: meshIndex,
                                                   mesh: mesh,
                                                   positionBuffer: posBuffer,
                                                   normalBuffer: nrmBuffer,
                                                   jointMatrixBuffer: jointMatrixBuffer))
                debugSkinnedIndex += 1
            }
        }

        guard !debugMeshes.isEmpty else {
            print("No debug buffers created.")
            return
        }

        let localResidencySet: MTLResidencySet? = {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.initialCapacity = debugMeshes.count * 8 + 16
            guard let set = try? device.makeResidencySet(descriptor: descriptor) else { return nil }

            var allocations: [any MTLAllocation] = []
            allocations.reserveCapacity(debugMeshes.count * 8 + 16)

            for debugMesh in debugMeshes {
                allocations.append(debugMesh.positionBuffer)
                allocations.append(debugMesh.normalBuffer)
                let buffers = debugMesh.mesh.mtkMesh.vertexBuffers
                if buffers.indices.contains(0) { allocations.append(buffers[0].buffer) }
                if buffers.indices.contains(1) { allocations.append(buffers[1].buffer) }
                if buffers.indices.contains(2) { allocations.append(buffers[2].buffer) }
                if buffers.indices.contains(3) { allocations.append(buffers[3].buffer) }
            }

            for buffer in skinningUniformBuffers {
                allocations.append(buffer)
            }
            for buffer in jointMatrixBuffers {
                allocations.append(buffer)
            }

            set.addAllocations(allocations)
            set.commit()
            return set
        }()

        guard let commandBuffer = device.makeCommandBuffer(),
              let allocator = device.makeCommandAllocator() else {
            print("Failed to create MTL4 command buffer/allocator.")
            return
        }

        commandBuffer.label = "Skinning Debug"
        commandBuffer.beginCommandBuffer(allocator: allocator)
        if let residencySet = commandQueueResidencySet {
            commandBuffer.useResidencySet(residencySet)
        }
        if let localResidencySet {
            commandBuffer.useResidencySet(localResidencySet)
        }

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create compute encoder for skinning debug.")
            return
        }

        computeEncoder.label = "Skinning Debug Encoder"
        computeEncoder.setComputePipelineState(skinningPipelineState)

        var tempUniformBuffers: [MTLBuffer] = []
        tempUniformBuffers.reserveCapacity(debugMeshes.count)

        var skinnedIndex = 0
        for (modelIndex, model) in scene.models.enumerated() {
            for mesh in model.meshes where mesh.hasSkinning {
                guard skinnedIndex < debugMeshes.count else { break }
                let debugMesh = debugMeshes[skinnedIndex]
                let vertexCount = mesh.mtkMesh.vertexCount
                if mesh.mtkMesh.vertexBuffers.count <= 3 {
                    print("[model \(modelIndex) mesh \(debugMesh.meshIndex)] Missing joint index/weight buffers. Skipping GPU dispatch.")
                    skinnedIndex += 1
                    continue
                }

                updateSkinningJointMatrices(model: model, mesh: mesh, destinationBuffer: debugMesh.jointMatrixBuffer)
                skinningTable.setAddress(debugMesh.jointMatrixBuffer.gpuAddress, index: BufferIndex.jointMatrices.rawValue)

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

                skinningTable.setAddress(debugMesh.positionBuffer.gpuAddress, index: BufferIndex.skinnedPositions.rawValue)
                skinningTable.setAddress(debugMesh.normalBuffer.gpuAddress, index: BufferIndex.skinnedNormals.rawValue)

                if skinnedIndex < skinningUniformBuffers.count {
                    skinningTable.setAddress(skinningUniformBuffers[skinnedIndex].gpuAddress, index: BufferIndex.uniforms.rawValue)
                } else {
                    var vertexCountU32 = UInt32(vertexCount)
                    if let uniformBuffer = device.makeBuffer(bytes: &vertexCountU32,
                                                             length: MemoryLayout<UInt32>.stride,
                                                             options: .storageModeShared) {
                        tempUniformBuffers.append(uniformBuffer)
                        skinningTable.setAddress(uniformBuffer.gpuAddress, index: BufferIndex.uniforms.rawValue)
                        if let localResidencySet {
                            localResidencySet.addAllocations([uniformBuffer])
                            localResidencySet.commit()
                        }
                    }
                }

                computeEncoder.setArgumentTable(skinningTable)

                let threadsPerGrid = MTLSize(width: Int(vertexCount), height: 1, depth: 1)
                let threadsPerGroup = MTLSize(width: min(Int(vertexCount), skinningPipelineState.maxTotalThreadsPerThreadgroup),
                                              height: 1,
                                              depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

                skinnedIndex += 1
            }
        }

        computeEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        commandQueue.commit([commandBuffer])

        let event = device.makeSharedEvent()!
        event.signaledValue = 0
        commandQueue.signalEvent(event, value: 1)
        while !event.wait(untilSignaledValue: 1, timeoutMS: 1000) {}

        var mergedOBJ = ""
        var mergedVertexOffset = 0
        var mergedNormalOffset = 0
        var mergedWroteFaces = false

        if let exportDirectory {
            do {
                try FileManager.default.createDirectory(at: exportDirectory, withIntermediateDirectories: true)
            } catch {
                print("Failed to create export directory: \(exportDirectory.path) error: \(error)")
            }
            mergedOBJ = "# Skinning CPU merged export\n"
        }

        for debugMesh in debugMeshes {
            let mesh = debugMesh.mesh
            let vertexCount = mesh.mtkMesh.vertexCount
            let safeName = mesh.name.isEmpty ? "mesh\(debugMesh.meshIndex)" : mesh.name
            let meshLabel = "model \(debugMesh.modelIndex) \(safeName)"
            guard mesh.mtkMesh.vertexBuffers.count > 3 else {
                print("[\(meshLabel)] Missing joint index/weight buffers. Skipping.")
                continue
            }

            let restPositions: [SIMD3<Float>] = SkinningDebug.readBuffer(buffer: mesh.mtkMesh.vertexBuffers[0].buffer,
                                                                         offset: mesh.mtkMesh.vertexBuffers[0].offset,
                                                                         count: vertexCount)
            let restNormals: [SIMD3<Float>] = SkinningDebug.readBuffer(buffer: mesh.mtkMesh.vertexBuffers[1].buffer,
                                                                       offset: mesh.mtkMesh.vertexBuffers[1].offset,
                                                                       count: vertexCount)
            let jointIndices: [SIMD4<UInt16>] = SkinningDebug.readBuffer(buffer: mesh.mtkMesh.vertexBuffers[2].buffer,
                                                                          offset: mesh.mtkMesh.vertexBuffers[2].offset,
                                                                          count: vertexCount)
            let jointWeights: [SIMD4<Float>] = SkinningDebug.readBuffer(buffer: mesh.mtkMesh.vertexBuffers[3].buffer,
                                                                        offset: mesh.mtkMesh.vertexBuffers[3].offset,
                                                                        count: vertexCount)

            if restPositions.isEmpty || restNormals.isEmpty || jointIndices.isEmpty || jointWeights.isEmpty {
                print("[\(meshLabel)] Missing CPU-readable skinning buffers. Skipping.")
                continue
            }

            let matrixBuffer = debugMesh.jointMatrixBuffer
            let matrixCount = matrixBuffer.length / MemoryLayout<matrix_float4x4>.stride
            let matrices: [matrix_float4x4] = SkinningDebug.readBuffer(buffer: matrixBuffer, offset: 0, count: matrixCount)

            let zeroMatrixCount = SkinningDebug.countZeroMatrices(matrices)
            if zeroMatrixCount > 0 {
                print("[\(meshLabel)] Joint matrices: \(zeroMatrixCount)/\(matrices.count) are zero matrices")
            }

            let cpuResult = SkinningDebug.cpuSkinning(vertexCount: vertexCount,
                                                      restPositions: restPositions,
                                                      restNormals: restNormals,
                                                      jointIndices: jointIndices,
                                                      jointWeights: jointWeights,
                                                      jointMatrices: matrices)

            if let exportDirectory {
                SkinningDebug.exportOBJ(modelIndex: debugMesh.modelIndex,
                                        meshIndex: debugMesh.meshIndex,
                                        mesh: mesh,
                                        positions: cpuResult.positions,
                                        normals: cpuResult.normals,
                                        to: exportDirectory)
                SkinningDebug.appendToMergedOBJ(output: &mergedOBJ,
                                                modelIndex: debugMesh.modelIndex,
                                                meshIndex: debugMesh.meshIndex,
                                                mesh: mesh,
                                                positions: cpuResult.positions,
                                                normals: cpuResult.normals,
                                                vertexOffset: &mergedVertexOffset,
                                                normalOffset: &mergedNormalOffset,
                                                wroteFaces: &mergedWroteFaces)
            }

            let gpuPositions: [SIMD3<Float>] = SkinningDebug.readBuffer(buffer: debugMesh.positionBuffer, offset: 0, count: vertexCount)
            let gpuNormals: [SIMD3<Float>] = SkinningDebug.readBuffer(buffer: debugMesh.normalBuffer, offset: 0, count: vertexCount)

            if gpuPositions.isEmpty || gpuNormals.isEmpty {
                print("[\(meshLabel)] GPU debug output not readable.")
                continue
            }

            let cpuZeroPos = SkinningDebug.countZeroVectors(cpuResult.positions, epsilon: epsilon)
            let cpuZeroNrm = SkinningDebug.countZeroVectors(cpuResult.normals, epsilon: epsilon)
            let cpuInvalid = SkinningDebug.countInvalidVectors(cpuResult.positions) + SkinningDebug.countInvalidVectors(cpuResult.normals)

            let gpuZeroPos = SkinningDebug.countZeroVectors(gpuPositions, epsilon: epsilon)
            let gpuZeroNrm = SkinningDebug.countZeroVectors(gpuNormals, epsilon: epsilon)
            let gpuInvalid = SkinningDebug.countInvalidVectors(gpuPositions) + SkinningDebug.countInvalidVectors(gpuNormals)

            print("[\(meshLabel)] Vertices: \(vertexCount)")
            print("[\(meshLabel)] CPU zeros pos/nrm: \(cpuZeroPos)/\(cpuZeroNrm), invalid: \(cpuInvalid)")
            print("[\(meshLabel)] GPU zeros pos/nrm: \(gpuZeroPos)/\(gpuZeroNrm), invalid: \(gpuInvalid)")
            print("[\(meshLabel)] Weight sum zero: \(cpuResult.stats.zeroWeightCount), sum!=1: \(cpuResult.stats.weightSumOutOfRangeCount), out-of-range joints: \(cpuResult.stats.outOfRangeJointIndexCount)")
            print("[\(meshLabel)] Weight range: min=\(cpuResult.stats.minWeight), max=\(cpuResult.stats.maxWeight), abnormal=\(cpuResult.stats.abnormalWeightCount)")

            let compareCount = min(cpuResult.positions.count, gpuPositions.count)
            var posErrors = 0
            var nrmErrors = 0

            for i in 0..<compareCount {
                let posDiff = simd_length(cpuResult.positions[i] - gpuPositions[i])
                if posDiff > epsilon {
                    posErrors += 1
                    if posErrors <= logLimit {
                        print("[\(meshLabel)] POS mismatch v\(i) CPU=\(cpuResult.positions[i]) GPU=\(gpuPositions[i]) Rest=\(restPositions[i])")
                        print("[\(meshLabel)]   Indices=\(jointIndices[i]) Weights=\(jointWeights[i])")
                    }
                }
                let nrmDiff = simd_length(cpuResult.normals[i] - gpuNormals[i])
                if nrmDiff > epsilon {
                    nrmErrors += 1
                }
            }

            if posErrors == 0 && nrmErrors == 0 {
                print("[\(meshLabel)] PASS: CPU and GPU match (epsilon \(epsilon))")
            } else {
                print("[\(meshLabel)] FAIL: \(posErrors) position mismatches, \(nrmErrors) normal mismatches")
                print("[\(meshLabel)] FAIL: \(posErrors) position mismatches, \(nrmErrors) normal mismatches")
            }
            
            // Geometric Consistency Check
            SkinningDebug.checkGeometricConsistency(mesh: mesh, positions: cpuResult.positions, normals: cpuResult.normals)
        }

        if let exportDirectory {
            SkinningDebug.exportMergedOBJ(output: mergedOBJ, wroteFaces: mergedWroteFaces, to: exportDirectory)
        }

        print("=== Skinning Debug Tests Done ===")
    }

    @discardableResult
    func runSkinningDebugTestsExportingToTmp(epsilon: Float = 0.001, logLimit: Int = 5) -> URL? {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("skinning_cpu_export_\(UUID().uuidString)", isDirectory: true)
        runSkinningDebugTests(epsilon: epsilon, logLimit: logLimit, exportDirectory: tmpDir)
        return tmpDir
    }
}

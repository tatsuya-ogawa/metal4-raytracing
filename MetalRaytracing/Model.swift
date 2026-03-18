//
//  Model.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit
import ModelIO
import ObjectiveC.runtime

struct ModelMaterialOverride {
    var baseColor: SIMD3<Float>?
    var refractionIndex: Float?
    var opacity: Float?

    init(baseColor: SIMD3<Float>? = nil, refractionIndex: Float? = nil, opacity: Float? = nil) {
        self.baseColor = baseColor
        self.refractionIndex = refractionIndex
        self.opacity = opacity
    }

    static func glass(tint: SIMD3<Float> = SIMD3<Float>(0.95, 0.98, 1.0),
                      refractionIndex: Float = 1.52,
                      opacity: Float = 0.08) -> ModelMaterialOverride {
        ModelMaterialOverride(baseColor: tint, refractionIndex: refractionIndex, opacity: opacity)
    }
}

class Model {
    var meshes: [Mesh]
    var skeleton: Skeleton?
    var animation: AnimationClip?
    
    // Animation state
    var currentTime: TimeInterval = 0
    var worldTransform: matrix_float4x4 = matrix_identity_float4x4
    
    // Skinning state
    var jointMatrices: [matrix_float4x4] = []
    
    var position: SIMD3<Float>
    var rotation: SIMD3<Float>
    var scale: Float
    
    init(name: String,
         position: SIMD3<Float>,
         rotation: SIMD3<Float> = [0, 0, 0],
         scale: Float,
         materialOverride: ModelMaterialOverride? = nil,
         on device: MTLDevice) {
        self.position = position
        self.rotation = rotation
        self.scale = scale
        
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        let scaleMatrix = matrix_float4x4.scale(scale)
        let translationMatrix = matrix_float4x4.translate(position)
        self.worldTransform = translationMatrix * rotationMatrix * scaleMatrix
        
        let allocator = MTKMeshBufferAllocator(device: device)
        
        // Try loading USDZ first, then OBJ
        var assetURL = Bundle.main.url(forResource: "AssetResources/\(name)", withExtension: "usdz")
        let isUSDZ = assetURL != nil
        if assetURL == nil {
            assetURL = Bundle.main.url(forResource: "AssetResources/\(name)", withExtension: "obj")
        }
        guard let url = assetURL else {
            fatalError("Model \(name) not found")
        }
        
        // For OBJ files, use the vertex descriptor at load time (original behavior)
        // For USDZ files, load without descriptor first, then apply later
        let asset: MDLAsset
        if isUSDZ {
            asset = MDLAsset(url: url, vertexDescriptor: Model.vertexDescriptor, bufferAllocator: allocator)
        } else {
            // OBJ: Use original loading method with vertex descriptor
            asset = MDLAsset(url: url, vertexDescriptor: Model.vertexDescriptor, bufferAllocator: allocator)
        }
        asset.loadTextures()
        
        // Initialize properties
        self.meshes = []
        self.jointMatrices = []
        
        if isUSDZ {
            // USDZ: Complex loading with skeleton/animation support
            let descriptor = Model.vertexDescriptor
            
            // Traverse to find skeleton and animation
            var foundSkeleton: MDLSkeleton?
            var foundAnimation: MDLPackedJointAnimation?
            
            func traverseAndFind(_ object: MDLObject) {
                if let skeleton = object as? MDLSkeleton {
                    foundSkeleton = skeleton
                }
                if let animation = object as? MDLPackedJointAnimation {
                    foundAnimation = animation
                }
                if let mesh = object as? MDLMesh {
                    if let bind = mesh.components.first(where: { $0 is MDLAnimationBindComponent }) as? MDLAnimationBindComponent {
                        if foundSkeleton == nil { foundSkeleton = bind.skeleton }
                        if foundAnimation == nil { foundAnimation = bind.jointAnimation as? MDLPackedJointAnimation }
                    }
                }
                for child in object.children.objects {
                    traverseAndFind(child)
                }
            }
            for i in 0..<asset.count {
                traverseAndFind(asset.object(at: i))
            }
            if foundAnimation == nil {
                for obj in asset.animations.objects {
                    if let anim = obj as? MDLPackedJointAnimation {
                        foundAnimation = anim
                        break
                    }
                }
            }
            
            // Initialize Skeleton
            if let mdlSkeleton = foundSkeleton {
                self.skeleton = Skeleton(from: mdlSkeleton)
            }
            
            // Initialize Animation
            if let mdlAnim = foundAnimation {
                self.animation = AnimationClip(from: mdlAnim)
            }
            
            // Helper to collect meshes (USDZ)
            func traverseAndCreateMesh(_ object: MDLObject) {
                if let mdlMesh = object as? MDLMesh {
                    if mdlMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal) == nil {
                        mdlMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.0)
                    }
                    if mdlMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTangent) == nil &&
                       mdlMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTextureCoordinate) != nil {
                        mdlMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                                normalAttributeNamed: MDLVertexAttributeNormal,
                                                tangentAttributeNamed: MDLVertexAttributeTangent)
                    }
                    
                    mdlMesh.vertexDescriptor = descriptor
                    
                    let mtkMesh = try! MTKMesh(mesh: mdlMesh, device: device)
                    
                    let bindComponent = mdlMesh.components.first(where: { $0 is MDLAnimationBindComponent }) as? MDLAnimationBindComponent
                    var skinningInfo: MeshSkinningInfo? = nil
                    if let bindComponent, let skeleton = self.skeleton {
                        let rawJointPaths = bindComponent.jointPaths ?? []
                        let jointPaths = rawJointPaths.isEmpty ? skeleton.jointPaths : rawJointPaths
                        let geometryBindTransform = matrix4x4_from_double(bindComponent.geometryBindTransform)
                        let geometryBindTransformInverse = simd_inverse(geometryBindTransform)
                        let pathToIndex = buildPathIndexMap(from: skeleton.jointPaths)
                        let tailToIndex = buildTailIndexMap(from: skeleton.jointPaths)
                        let jointToSkeletonIndex = jointPaths.map { mapJointPathToSkeletonIndex($0, pathToIndex: pathToIndex, tailToIndex: tailToIndex) }
                        skinningInfo = MeshSkinningInfo(jointPaths: jointPaths,
                                                        jointToSkeletonIndex: jointToSkeletonIndex,
                                                        geometryBindTransform: geometryBindTransform,
                                                        geometryBindTransformInverse: geometryBindTransformInverse)
                    }
                    
                    let mesh = Mesh(modelName: name,
                                    mdlMesh: mdlMesh,
                                    mtkMesh: mtkMesh,
                                    transform: self.worldTransform,
                                    hasSkinning: skinningInfo != nil,
                                    skinning: skinningInfo,
                                    on: device)
                    self.meshes.append(mesh)
                }
                for child in object.children.objects {
                    traverseAndCreateMesh(child)
                }
            }
            for i in 0..<asset.count {
                traverseAndCreateMesh(asset.object(at: i))
            }
            
            update(deltaTime: 0)
        } else {
            // OBJ: Simple loading (original behavior)
            let mdlMeshes = asset.childObjects(of: MDLMesh.self) as! [MDLMesh]
            
            self.meshes = mdlMeshes.map { mdlMesh -> Mesh in
                let mtkMesh = try! MTKMesh(mesh: mdlMesh, device: device)
                return Mesh(modelName: name, mdlMesh: mdlMesh, mtkMesh: mtkMesh, position: position, rotation: rotation, scale: scale, on: device)
            }
        }

        applyMaterialOverride(materialOverride)
    }

    private func applyMaterialOverride(_ materialOverride: ModelMaterialOverride?) {
        guard let materialOverride else { return }
        for mesh in meshes {
            for submesh in mesh.submeshes {
                submesh.applyMaterialOverride(materialOverride)
            }
        }
    }
    
    func update(deltaTime: TimeInterval) {
        // Update animation time
        if let animation = animation {
            let duration = animation.duration
            if duration > 0 {
                currentTime += deltaTime
                currentTime = fmod(currentTime, duration)
            }
        }
        
        guard let skeleton = skeleton else { return }
        
        // Compute local transforms
        var localTransforms = skeleton.restTransforms
        
        if let animation = animation {
            let (translations, rotations, scales) = animation.sample(at: currentTime)
            let animCount = min(translations.count,
                                min(rotations.count,
                                    min(scales.count, animation.jointPaths.count)))
            let pathToIndex = buildPathIndexMap(from: skeleton.jointPaths)
            let tailToIndex = buildTailIndexMap(from: skeleton.jointPaths)
            
            for i in 0..<animCount {
                let jointIndex = mapJointPathToSkeletonIndex(animation.jointPaths[i],
                                                             pathToIndex: pathToIndex,
                                                             tailToIndex: tailToIndex)
                guard jointIndex >= 0 && jointIndex < localTransforms.count else { continue }
                
                var rotation = rotations[i]
                let qLength = sqrt(rotation.real * rotation.real +
                                  rotation.imag.x * rotation.imag.x +
                                  rotation.imag.y * rotation.imag.y +
                                  rotation.imag.z * rotation.imag.z)
                if qLength > 0.0001 {
                    rotation = simd_quatf(ix: rotation.imag.x / qLength,
                                          iy: rotation.imag.y / qLength,
                                          iz: rotation.imag.z / qLength,
                                          r: rotation.real / qLength)
                } else {
                    rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
                }
                
                localTransforms[jointIndex] = matrix4x4_trs(translation: translations[i],
                                                            rotation: rotation,
                                                            scale: scales[i])
            }
        }
        
        // Compute global transforms
        let globalTransforms = skeleton.computeGlobalTransforms(localTransforms: localTransforms)
        
        // Compute skinning matrices (Global * InverseBind)
        jointMatrices = zip(globalTransforms, skeleton.inverseBindTransforms).map { $0 * $1 }
    }
    
    func updateTransform() {
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        let scaleMatrix = matrix_float4x4.scale(scale)
        let translationMatrix = matrix_float4x4.translate(position)
        self.worldTransform = translationMatrix * rotationMatrix * scaleMatrix
        
        for i in 0..<meshes.count {
            meshes[i].transform = self.worldTransform
        }
    }
    
    // Movement helpers
    func forward(direction: Float) {
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        // Assuming -Z is forward in local space, or standard conventions
        // Let's assume standard OpenGL/Metal: +Y up, -Z forward, +X right
        let forwardVector = SIMD3<Float>(0, 0, -1)
        let localForward = simd_make_float3(rotationMatrix * simd_make_float4(forwardVector.x, forwardVector.y, forwardVector.z, 0))
        
        position += normalize(localForward) * direction
        updateTransform()
    }
    
    func strafe(direction: Float) {
        let rotationMatrix = matrix_float4x4.rotate(rotation)
        let rightVector = SIMD3<Float>(1, 0, 0)
        let localRight = simd_make_float3(rotationMatrix * simd_make_float4(rightVector.x, rightVector.y, rightVector.z, 0))
        position += normalize(localRight) * direction
        updateTransform()
    }

    func rotateY(angle: Float) {
        rotation.y += angle
        updateTransform()
    }
    
    func setRotationY(angle: Float) {
        rotation.y = angle
        updateTransform()
    }
    
    static var vertexDescriptor: MDLVertexDescriptor = {
        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.attributes[0] =
        MDLVertexAttribute(name: MDLVertexAttributePosition,
                           format: .float3,
                           offset: 0, bufferIndex: 0)
        vertexDescriptor.attributes[1] =
        MDLVertexAttribute(name: MDLVertexAttributeNormal,
                           format: .float3,
                           offset: 0, bufferIndex: 1)
        
        // Add Skinning Attributes
        let jointIndicesAttr = MDLVertexAttribute(name: MDLVertexAttributeJointIndices,
                           format: .uShort4,
                           offset: 0, bufferIndex: 2)
        jointIndicesAttr.initializationValue = vector_float4(0, 0, 0, 0)
        vertexDescriptor.attributes[2] = jointIndicesAttr
        
        let jointWeightsAttr = MDLVertexAttribute(name: MDLVertexAttributeJointWeights,
                           format: .float4,
                           offset: 0, bufferIndex: 3)
        jointWeightsAttr.initializationValue = vector_float4(1, 0, 0, 0)
        vertexDescriptor.attributes[3] = jointWeightsAttr
        
        // Add Texture Coordinate Attribute
        let textureCoordinateAttr = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate,
                                                    format: .float2,
                                                    offset: 0, bufferIndex: 4)
        textureCoordinateAttr.initializationValue = vector_float4(0, 0, 0, 0)
        vertexDescriptor.attributes[4] = textureCoordinateAttr
        
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        vertexDescriptor.layouts[1] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        vertexDescriptor.layouts[2] = MDLVertexBufferLayout(stride: MemoryLayout<UInt16>.stride * 4)
        vertexDescriptor.layouts[3] = MDLVertexBufferLayout(stride: MemoryLayout<Float>.stride * 4)
        vertexDescriptor.layouts[4] = MDLVertexBufferLayout(stride: MemoryLayout<SIMD2<Float>>.stride)
        return vertexDescriptor
    }()
}

// MARK: - Helper Classes

class Skeleton {
    let jointPaths: [String]
    let parentIndices: [Int]
    let restTransforms: [matrix_float4x4]
    let inverseBindTransforms: [matrix_float4x4]
    
    init(from mdlSkeleton: MDLSkeleton) {
        // Build parent indices from paths
        let paths = mdlSkeleton.jointPaths
        self.jointPaths = paths

        let pathToIndex = buildPathIndexMap(from: paths)
        self.parentIndices = paths.map { path in
            guard let parentPath = parentJointPath(for: path),
                  let parentIndex = pathToIndex[parentPath] else {
                return -1
            }
            return parentIndex
        }
        
        if let binds = mdlSkeleton.jointBindTransforms.float4x4Array as? [matrix_float4x4] {
             self.inverseBindTransforms = binds.map { $0.inverse }
        } else {
             self.inverseBindTransforms = Array(repeating: matrix_identity_float4x4, count: jointPaths.count)
        }
        
        if let rests = mdlSkeleton.jointRestTransforms.float4x4Array as? [matrix_float4x4] { // Cast to ensure type match
            self.restTransforms = rests
        } else {
            self.restTransforms = Array(repeating: matrix_identity_float4x4, count: jointPaths.count)
        }
    }
    
    func computeGlobalTransforms(localTransforms: [matrix_float4x4]) -> [matrix_float4x4] {
        guard localTransforms.count == parentIndices.count else { return localTransforms }

        var globals = Array(repeating: matrix_identity_float4x4, count: localTransforms.count)
        var resolutionState = Array(repeating: UInt8(0), count: localTransforms.count)

        func resolveGlobalTransform(for jointIndex: Int) -> matrix_float4x4 {
            if resolutionState[jointIndex] == 2 {
                return globals[jointIndex]
            }
            if resolutionState[jointIndex] == 1 {
                // Break malformed parent cycles by treating the current local transform as authoritative.
                return localTransforms[jointIndex]
            }

            resolutionState[jointIndex] = 1
            let parentIndex = parentIndices[jointIndex]
            if parentIndex >= 0 && parentIndex < localTransforms.count {
                globals[jointIndex] = resolveGlobalTransform(for: parentIndex) * localTransforms[jointIndex]
            } else {
                globals[jointIndex] = localTransforms[jointIndex]
            }
            resolutionState[jointIndex] = 2
            return globals[jointIndex]
        }

        for jointIndex in localTransforms.indices {
            _ = resolveGlobalTransform(for: jointIndex)
        }
        return globals
    }
}

class AnimationClip {
    let startTime: TimeInterval
    let endTime: TimeInterval
    let duration: TimeInterval
    let translations: MDLAnimatedVector3Array
    let rotations: MDLAnimatedQuaternionArray
    let scales: MDLAnimatedVector3Array
    let jointPaths: [String]
    fileprivate let translationTrack: Float3AnimationTrack?
    fileprivate let rotationTrack: QuaternionAnimationTrack?
    fileprivate let scaleTrack: Float3AnimationTrack?
    
    init(from packed: MDLPackedJointAnimation) {
        self.translations = packed.translations
        self.rotations = packed.rotations
        self.scales = packed.scales
        self.jointPaths = packed.jointPaths
        self.translationTrack = Float3AnimationTrack(array: packed.translations)
        self.rotationTrack = QuaternionAnimationTrack(array: packed.rotations)
        self.scaleTrack = Float3AnimationTrack(array: packed.scales)
        
        let maxTime = max(translations.maximumTime, max(rotations.maximumTime, scales.maximumTime))
        let minTime = min(translations.minimumTime, min(rotations.minimumTime, scales.minimumTime))
        self.startTime = minTime
        self.endTime = maxTime
        self.duration = maxTime - minTime
    }
    
    func sample(at time: TimeInterval) -> ([SIMD3<Float>], [simd_quatf], [SIMD3<Float>]) {
        let sampleTime: TimeInterval
        if duration > 0 {
            let wrappedTime = time.truncatingRemainder(dividingBy: duration)
            let positiveWrappedTime = wrappedTime >= 0 ? wrappedTime : wrappedTime + duration
            sampleTime = startTime + positiveWrappedTime
        } else {
            sampleTime = startTime
        }

        let clampedSampleTime = min(max(sampleTime, startTime), endTime)
        let t = translationTrack?.sampleAll(at: clampedSampleTime) ?? translations.float3Array(atTime: clampedSampleTime)
        let r = rotationTrack?.sampleAll(at: clampedSampleTime) ?? rotations.floatQuaternionArray(atTime: clampedSampleTime)
        let s = scaleTrack?.sampleAll(at: clampedSampleTime) ?? scales.float3Array(atTime: clampedSampleTime)
        return (t, r, s)
    }
}

fileprivate struct Float3AnimationTrack {
    let elementCount: Int
    let keyTimes: [TimeInterval]
    let values: [SIMD3<Float>]

    init?(array: MDLAnimatedVector3Array) {
        self.elementCount = array.elementCount
        self.keyTimes = array.keyTimes.map(\.doubleValue)

        let sampleCount = max(1, max(array.timeSampleCount, keyTimes.count))
        let totalCount = elementCount * sampleCount
        guard totalCount > 0 else { return nil }

        var rawValues = Array(repeating: SIMD3<Float>(repeating: 0), count: totalCount)
        let copiedCount = rawValues.withUnsafeMutableBufferPointer { buffer -> Int in
            guard let baseAddress = buffer.baseAddress else { return 0 }
            return callFloat3ArrayGetter(object: array,
                                         selectorName: "getFloat3Array:maxCount:",
                                         output: baseAddress,
                                         maxCount: totalCount)
        }
        guard copiedCount >= elementCount else { return nil }

        if copiedCount == totalCount {
            self.values = rawValues
        } else {
            self.values = Array(rawValues.prefix(copiedCount))
        }
    }

    func sampleAll(at time: TimeInterval) -> [SIMD3<Float>] {
        (0..<elementCount).map { sample(elementAt: $0, time: time) }
    }

    private func sample(elementAt index: Int, time: TimeInterval) -> SIMD3<Float> {
        if values.isEmpty {
            return SIMD3<Float>(repeating: 0)
        }
        if keyTimes.count <= 1 {
            return values[safe: index] ?? SIMD3<Float>(repeating: 0)
        }

        let upperIndex = keyTimes.partitioningIndex { $0 > time }
        if upperIndex <= 0 {
            return value(timeIndex: 0, elementIndex: index)
        }
        if upperIndex >= keyTimes.count {
            return value(timeIndex: keyTimes.count - 1, elementIndex: index)
        }

        let lowerIndex = upperIndex - 1
        let t0 = keyTimes[lowerIndex]
        let t1 = keyTimes[upperIndex]
        let v0 = value(timeIndex: lowerIndex, elementIndex: index)
        let v1 = value(timeIndex: upperIndex, elementIndex: index)

        guard abs(t1 - t0) > 1.0e-8 else { return v0 }
        let alpha = Float((time - t0) / (t1 - t0))
        return simd_mix(v0, v1, SIMD3<Float>(repeating: alpha))
    }

    private func value(timeIndex: Int, elementIndex: Int) -> SIMD3<Float> {
        let flatIndex = timeIndex * elementCount + elementIndex
        if flatIndex >= 0 && flatIndex < values.count {
            return values[flatIndex]
        }
        return values[safe: elementIndex] ?? SIMD3<Float>(repeating: 0)
    }
}

fileprivate struct QuaternionAnimationTrack {
    let elementCount: Int
    let keyTimes: [TimeInterval]
    let values: [simd_quatf]

    init?(array: MDLAnimatedQuaternionArray) {
        self.elementCount = array.elementCount
        self.keyTimes = array.keyTimes.map(\.doubleValue)

        let sampleCount = max(1, max(array.timeSampleCount, keyTimes.count))
        let totalCount = elementCount * sampleCount
        guard totalCount > 0 else { return nil }

        var rawValues = Array(repeating: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), count: totalCount)
        let copiedCount = rawValues.withUnsafeMutableBufferPointer { buffer -> Int in
            guard let baseAddress = buffer.baseAddress else { return 0 }
            return callQuaternionArrayGetter(object: array,
                                             selectorName: "getFloatQuaternionArray:maxCount:",
                                             output: baseAddress,
                                             maxCount: totalCount)
        }
        guard copiedCount >= elementCount else { return nil }

        let effectiveValues = copiedCount == totalCount ? rawValues : Array(rawValues.prefix(copiedCount))
        self.values = effectiveValues.map { quaternion in
            let length = simd_length(quaternion.vector)
            guard length > 1.0e-8 else {
                return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            }
            return simd_quatf(vector: quaternion.vector / length)
        }
    }

    func sampleAll(at time: TimeInterval) -> [simd_quatf] {
        (0..<elementCount).map { sample(elementAt: $0, time: time) }
    }

    private func sample(elementAt index: Int, time: TimeInterval) -> simd_quatf {
        if values.isEmpty {
            return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        if keyTimes.count <= 1 {
            return values[safe: index] ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }

        let upperIndex = keyTimes.partitioningIndex { $0 > time }
        if upperIndex <= 0 {
            return value(timeIndex: 0, elementIndex: index)
        }
        if upperIndex >= keyTimes.count {
            return value(timeIndex: keyTimes.count - 1, elementIndex: index)
        }

        let lowerIndex = upperIndex - 1
        let t0 = keyTimes[lowerIndex]
        let t1 = keyTimes[upperIndex]
        let q0 = value(timeIndex: lowerIndex, elementIndex: index)
        var q1 = value(timeIndex: upperIndex, elementIndex: index)
        if simd_dot(q0.vector, q1.vector) < 0 {
            q1 = simd_quatf(vector: -q1.vector)
        }

        guard abs(t1 - t0) > 1.0e-8 else { return q0 }
        let alpha = Float((time - t0) / (t1 - t0))
        return simd_slerp(q0, q1, alpha)
    }

    private func value(timeIndex: Int, elementIndex: Int) -> simd_quatf {
        let flatIndex = timeIndex * elementCount + elementIndex
        if flatIndex >= 0 && flatIndex < values.count {
            return values[flatIndex]
        }
        return values[safe: elementIndex] ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }
}

// MARK: - Math Helpers

private func matrix4x4_from_double(_ m: simd_double4x4) -> matrix_float4x4 {
    return matrix_float4x4(columns: (
        SIMD4<Float>(Float(m.columns.0.x), Float(m.columns.0.y), Float(m.columns.0.z), Float(m.columns.0.w)),
        SIMD4<Float>(Float(m.columns.1.x), Float(m.columns.1.y), Float(m.columns.1.z), Float(m.columns.1.w)),
        SIMD4<Float>(Float(m.columns.2.x), Float(m.columns.2.y), Float(m.columns.2.z), Float(m.columns.2.w)),
        SIMD4<Float>(Float(m.columns.3.x), Float(m.columns.3.y), Float(m.columns.3.z), Float(m.columns.3.w))
    ))
}

private func normalizeJointPath(_ path: String) -> String {
    let parts = path.split(separator: "/").filter { !$0.isEmpty }
    return parts.joined(separator: "/")
}

private func parentJointPath(for path: String) -> String? {
    let normalized = normalizeJointPath(path)
    guard let lastSlash = normalized.lastIndex(of: "/") else { return nil }
    let parent = String(normalized[..<lastSlash])
    return parent.isEmpty ? nil : parent
}

private func buildPathIndexMap(from jointPaths: [String]) -> [String: Int] {
    let normalizedPaths = jointPaths.map { normalizeJointPath($0) }
    var map: [String: Int] = [:]
    for (index, path) in normalizedPaths.enumerated() where !path.isEmpty {
        map[path] = index
    }
    
    var suffixCounts: [String: Int] = [:]
    for path in normalizedPaths where !path.isEmpty {
        let parts = path.split(separator: "/")
        guard parts.count > 1 else { continue }
        for start in 1..<parts.count {
            let suffix = parts[start...].joined(separator: "/")
            suffixCounts[suffix, default: 0] += 1
        }
    }
    
    for (index, path) in normalizedPaths.enumerated() where !path.isEmpty {
        let parts = path.split(separator: "/")
        guard parts.count > 1 else { continue }
        for start in 1..<parts.count {
            let suffix = parts[start...].joined(separator: "/")
            if suffixCounts[suffix] == 1 && map[suffix] == nil {
                map[suffix] = index
            }
        }
    }
    return map
}

private func buildTailIndexMap(from jointPaths: [String]) -> [String: Int] {
    var counts: [String: Int] = [:]
    let tails = jointPaths.map { path -> String in
        let normalized = normalizeJointPath(path)
        return normalized.split(separator: "/").last.map(String.init) ?? normalized
    }
    for tail in tails where !tail.isEmpty {
        counts[tail, default: 0] += 1
    }
    var map: [String: Int] = [:]
    for (index, tail) in tails.enumerated() where !tail.isEmpty {
        if counts[tail] == 1 {
            map[tail] = index
        }
    }
    return map
}

private func mapJointPathToSkeletonIndex(_ jointPath: String,
                                         pathToIndex: [String: Int],
                                         tailToIndex: [String: Int]) -> Int {
    let normalized = normalizeJointPath(jointPath)
    if let index = pathToIndex[normalized] {
        return index
    }
    let tail = normalized.split(separator: "/").last.map(String.init) ?? normalized
    if let index = tailToIndex[tail] {
        return index
    }
    return -1
}

func matrix4x4_trs(translation: SIMD3<Float>, rotation: simd_quatf, scale: SIMD3<Float>) -> matrix_float4x4 {
    let translationMatrix = matrix_float4x4.translate(translation)
    let rotationMatrix = matrix_float4x4(rotation)
    let scaleMatrix = matrix_float4x4.scale(scale)
    return translationMatrix * rotationMatrix * scaleMatrix
}

private func callFloat3ArrayGetter(object: NSObject,
                                   selectorName: String,
                                   output: UnsafeMutablePointer<SIMD3<Float>>,
                                   maxCount: Int) -> Int {
    let selector = NSSelectorFromString(selectorName)
    guard object.responds(to: selector) else { return 0 }
    typealias Getter = @convention(c) (AnyObject, Selector, UnsafeMutablePointer<SIMD3<Float>>, Int) -> Int
    let implementation = object.method(for: selector)
    let function = unsafeBitCast(implementation, to: Getter.self)
    return function(object, selector, output, maxCount)
}

private func callQuaternionArrayGetter(object: NSObject,
                                       selectorName: String,
                                       output: UnsafeMutablePointer<simd_quatf>,
                                       maxCount: Int) -> Int {
    let selector = NSSelectorFromString(selectorName)
    guard object.responds(to: selector) else { return 0 }
    typealias Getter = @convention(c) (AnyObject, Selector, UnsafeMutablePointer<simd_quatf>, Int) -> Int
    let implementation = object.method(for: selector)
    let function = unsafeBitCast(implementation, to: Getter.self)
    return function(object, selector, output, maxCount)
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}

private extension RandomAccessCollection {
    func partitioningIndex(where predicate: (Element) -> Bool) -> Index {
        var low = startIndex
        var high = endIndex
        while low != high {
            let distance = self.distance(from: low, to: high)
            let mid = index(low, offsetBy: distance / 2)
            if predicate(self[mid]) {
                high = mid
            } else {
                low = index(after: mid)
            }
        }
        return low
    }
}

// Extensions moved to Utilities.swift or use existing ones

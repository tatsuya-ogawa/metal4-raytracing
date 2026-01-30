//
//  Model.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit
import ModelIO

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
    
    init(name: String, position: SIMD3<Float>, rotation: SIMD3<Float> = [0, 0, 0], scale: Float, on device: MTLDevice) {
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
        var globals = localTransforms
        for (i, parentIndex) in parentIndices.enumerated() {
            if parentIndex >= 0 && parentIndex < i { // Assumption: parents always come before children
                globals[i] = globals[parentIndex] * localTransforms[i]
            }
        }
        return globals
    }
}

class AnimationClip {
    let duration: TimeInterval
    let translations: MDLAnimatedVector3Array
    let rotations: MDLAnimatedQuaternionArray
    let scales: MDLAnimatedVector3Array
    let jointPaths: [String]
    
    init(from packed: MDLPackedJointAnimation) {
        self.translations = packed.translations
        self.rotations = packed.rotations
        self.scales = packed.scales
        self.jointPaths = packed.jointPaths
        
        let maxTime = max(translations.maximumTime, max(rotations.maximumTime, scales.maximumTime))
        let minTime = min(translations.minimumTime, min(rotations.minimumTime, scales.minimumTime))
        self.duration = maxTime - minTime
    }
    
    func sample(at time: TimeInterval) -> ([SIMD3<Float>], [simd_quatf], [SIMD3<Float>]) {
        let t = translations.float3Array(atTime: time)
        let r = rotations.floatQuaternionArray(atTime: time)
        let s = scales.float3Array(atTime: time)
        return (t, r, s)
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

// Extensions moved to Utilities.swift or use existing ones

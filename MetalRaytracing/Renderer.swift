//
//  Renderer.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//  Updated by Tatsuya Ogawa on 26/01/2026
//

import Metal
import MetalKit
import MetalFX
import simd

enum ShadingMode: Int {
    case pbr = 0
    case legacy = 1
}

class Renderer: NSObject {
    
    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let legacyCommandQueue: MTLCommandQueue // Add this for legacy AS building
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let library: MTLLibrary
    let computeTable: MTL4ArgumentTable
    var renderTable: MTL4ArgumentTable!
    var renderTableDesc: MTL4ArgumentTableDescriptor!
    
    // Skinning
    var skinningPass: SkinningPass!
    
    // We need to rebuild BLAS for these meshes
    // MTL4 AS
    var commandQueueResidencySet: MTLResidencySet?
    var scratchBuffer: MTLBuffer?
    
    // Initial creation flag
    var useFastBuild = true
    let endFrameEvent: MTLSharedEvent
    let computeEvent: MTLSharedEvent
    var gpuFrameIndex: UInt64 = 0
    
    var scene: Scene
    
    var raytracingPipelineState: MTLComputePipelineState!
    var renderPipelineState: MTLRenderPipelineState!
    
    var accumulationTargets: [MTLTexture] = []
    var randomTexture: MTLTexture!
    var renderScale: Float = 0.67 { // 0.67 gives better quality/performance balance than 0.5
        didSet {
            renderScale = max(0.25, min(1.0, renderScale))
            if lastDrawableSize.width > 0 && lastDrawableSize.height > 0 {
                createTextures(outputSize: lastDrawableSize)
                frameIndex = 0
            }
        }
    }
    private var depthTexture: MTLTexture?
    private var motionTexture: MTLTexture?
    private var diffuseAlbedoTexture: MTLTexture?
    private var specularAlbedoTexture: MTLTexture?
    private var normalTexture: MTLTexture?
    private var roughnessTexture: MTLTexture?
    private var metal4Compiler: MTL4Compiler?
    
    var framePresenter: FramePresenter!
    
    // Temporal scaler toggle
    var useTemporalScaler: Bool = false {
        didSet {
            if oldValue != useTemporalScaler {
                frameIndex = 0
                framePresenter.useTemporalScaler = useTemporalScaler
                
                if lastDrawableSize.width > 0 && lastDrawableSize.height > 0 {
                    createTextures(outputSize: lastDrawableSize)
                }
            }
        }
    }
    
    var useTemporalDenoiser: Bool = false {
        didSet {
            if oldValue != useTemporalDenoiser {
                frameIndex = 0
                framePresenter.useTemporalDenoiser = useTemporalDenoiser
                
                if lastDrawableSize.width > 0 && lastDrawableSize.height > 0 {
                    createTextures(outputSize: lastDrawableSize)
                }
            }
        }
    }
    
    // Camera history for motion vectors
    private var previousCamera: Camera?
    
    // MetalFX toggle
    var isMetalFXEnabled: Bool = true {
        didSet {
            if oldValue != isMetalFXEnabled {
                frameIndex = 0
                framePresenter.isMetalFXEnabled = isMetalFXEnabled
                
                // Recreate textures with or without upscaling
                if lastDrawableSize.width > 0 && lastDrawableSize.height > 0 {
                    createTextures(outputSize: lastDrawableSize)
                }
            }
        }
    }
    
    // Samples per pixel (higher = less noise, lower framerate)
    var samplesPerPixel: Int32 = 2 {
        didSet {
            frameIndex = 0
        }
    }
    
    // Exponential moving average weight for accumulation (0 = no history)
    var accumulationWeight: Float = 0.9 {
        didSet {
            frameIndex = 0
        }
    }
    
    // Reduce history weight based on motion magnitude (per-pixel).
    var useMotionAdaptiveAccumulation: Bool = true {
        didSet {
            frameIndex = 0
        }
    }
    var motionAccumulationMinWeight: Float = 0.1 {
        didSet {
            frameIndex = 0
        }
    }
    var motionAccumulationLowThresholdPixels: Float = 0.5 {
        didSet {
            frameIndex = 0
        }
    }
    var motionAccumulationHighThresholdPixels: Float = 4.0 {
        didSet {
            frameIndex = 0
        }
    }
    
    
    // Max ray bounces (higher = more realistic lighting, slower)
    var maxBounces: Int32 = 2 {
        didSet {
            frameIndex = 0
        }
    }
    
    var debugTextureMode: Int32 = 0 {
        didSet {
            frameIndex = 0
        }
    }
    
    var shadingMode: ShadingMode = .pbr {
        didSet {
            frameIndex = 0
        }
    }
    
    func setLightIntensity(_ intensity: Float) {
        scene.setLightIntensity(intensity)
        frameIndex = 0
    }
    
    var uniformBuffer: MTLBuffer!
    var resourcesBuffer: MTLBuffer!
    var instanceDescriptorBuffer: MTLBuffer!
    var previousInstanceDescriptorBuffer: MTLBuffer!
    
    var instancedAccelarationStructure: MTLAccelerationStructure!
    var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
    
    let maxFramesInFlight = 3
    
    var uniformBufferIndex = 0
    var uniformBufferOffset: Int {
        uniformBufferIndex * MemoryLayout<Uniforms>.stride
    }
    
    var frameIndex: UInt32 = 0
    var resourceStride = 0
    var maxSubmeshes = 0
    private var lastDrawableSize = CGSize.zero
    
    var cameraTarget = SIMD3<Float>(0.0, 0.0, 0.0)
    var cameraAzimuth: Float = 0.0
    var cameraElevation: Float = 0.0
    var cameraDistance: Float = 5.0
    var cameraFovDegrees: Float = 45.0
    private let minCameraDistance: Float = 1.5
    private let maxCameraDistance: Float = 50.0
    private let cameraElevationLimit: Float = (Float.pi / 2.0) - 0.01
    
    init?(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("GPU not available")
        }
        
        let size = metalView.frame.size
        
        metalView.device = device
        metalView.colorPixelFormat = .rgba16Float // .rgba16Float
        metalView.sampleCount = 1
        metalView.drawableSize = size
        
        self.device = device
        self.commandQueue = device.makeMTL4CommandQueue()!
        self.legacyCommandQueue = device.makeCommandQueue()!
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0..<maxFramesInFlight).map { _ in device.makeCommandAllocator()! }
        self.library = device.makeDefaultLibrary()!
        
        let computeTableDesc = MTL4ArgumentTableDescriptor()
        computeTableDesc.maxBufferBindCount = 18
        computeTableDesc.maxTextureBindCount = 12
        self.computeTable = try! device.makeArgumentTable(descriptor: computeTableDesc)
        
        let renderTableDesc = MTL4ArgumentTableDescriptor()
        renderTableDesc.maxTextureBindCount = 1
        self.renderTable = try! device.makeArgumentTable(descriptor: renderTableDesc)
        
        // Skinning Table
        self.skinningPass = SkinningPass(device: device, library: library)
        
        self.endFrameEvent = device.makeSharedEvent()!
        self.computeEvent = device.makeSharedEvent()!
        self.gpuFrameIndex = UInt64(maxFramesInFlight)
        self.endFrameEvent.signaledValue = UInt64(maxFramesInFlight - 1)
        
        self.scene = AppScene(size: size, device: device)
        self.cameraTarget = scene.cameraTarget
        self.cameraAzimuth = scene.cameraAzimuth
        self.cameraElevation = scene.cameraElevation
        self.cameraDistance = scene.cameraDistance
        self.cameraFovDegrees = scene.cameraFovDegrees
        
        super.init()
        
        createBuffers()
        createPipelineStates(metalView: metalView)
        
        if canUseMTL4AccelerationStructures(device: device){
            self.framePresenter = MTL4UpScaleRenderer(device: device, library: library, endFrameEvent: endFrameEvent, commandQueue: self.commandQueue, metal4Compiler: self.metal4Compiler!)!
        }else{
            self.framePresenter = LegacyFramePresenter(renderer: LecacyUpScaleRenderer(device: device, library: library, commandQueue: legacyCommandQueue, endFrameEvent: endFrameEvent)!, commandQueue: self.commandQueue)
        }
        self.framePresenter.useTemporalScaler = useTemporalScaler
        self.framePresenter.useTemporalDenoiser = useTemporalDenoiser
        self.framePresenter.isMetalFXEnabled = isMetalFXEnabled
        
        // Set player model index if needed, e.g. find "Dragon" or just 0
        //        if !scene.models.isEmpty {
        //            playerModelIndex = 0 // Control the first model
        //        }
        
        if canUseMTL4AccelerationStructures(device: device){
            createMTL4AccelerationStructures()
        }else{
            createLegacyAccelerationStructures()
        }
        
        mtkView(metalView, drawableSizeWillChange: size)
        metalView.delegate = self
    }
    func canUseMTL4AccelerationStructures(device:MTLDevice)->Bool{
        return device.supportsFamily(MTLGPUFamily.apple9)
    }
    
    func createPipelineStates(metalView: MTKView) {
        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        computeDescriptor.label = "Raytracing PipelineState"
        
        let functionConstants = MTLFunctionConstantValues()
        var resourceStride = Int32(self.resourceStride)
        functionConstants.setConstantValue(&resourceStride, type: .int, index: 0)
        var maxSubmeshes = Int32(self.maxSubmeshes)
        functionConstants.setConstantValue(&maxSubmeshes, type: .int, index: 1)
        
        do {
            computeDescriptor.computeFunction = try library.makeFunction(name: "raytracingKernel", constantValues: functionConstants)
            
            raytracingPipelineState = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            
            let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
            metal4Compiler = compiler
            
            let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
            vertexFunctionDescriptor.library = library
            vertexFunctionDescriptor.name = "vertexShader"
            
            let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
            fragmentFunctionDescriptor.library = library
            fragmentFunctionDescriptor.name = "fragmentShader"
            
            let pipelineDescriptor = MTL4RenderPipelineDescriptor()
            pipelineDescriptor.rasterSampleCount = metalView.sampleCount
            pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDescriptor
            pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDescriptor
            pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
            pipelineDescriptor.label = "Render PipelineState"
            
            renderPipelineState = try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print(error.localizedDescription)
        }
    }
    func createBuffers() {
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride * maxFramesInFlight, options: CommonStorageMode.options)!
        uniformBuffer.label = "Uniform Buffer"
        
        self.resourceStride = 0
        for model in scene.models {
            for mesh in model.meshes {
                for submesh in mesh.submeshes {
                    let encoder = device.makeArgumentEncoder(for: submesh.resources)!
                    if encoder.encodedLength > resourceStride {
                        resourceStride = encoder.encodedLength
                    }
                }
            }
        }
        self.maxSubmeshes = scene.models.flatMap(\.meshes).map { $0.submeshes.count }.max()!
        let resourceCount = maxSubmeshes * scene.models.flatMap(\.meshes).count
        
        resourcesBuffer = device.makeBuffer(length: resourceStride * resourceCount/*scene.geometries.count*/, options: CommonStorageMode.options)!
        resourcesBuffer.label = "Resources Buffer"
        
        for (i, mesh) in scene.models.flatMap(\.meshes).enumerated() {
            for (j, submesh) in mesh.submeshes.enumerated() {
                let index = i * maxSubmeshes + j
                let encoder = device.makeArgumentEncoder(for: submesh.resources)!
                encoder.setArgumentBuffer(resourcesBuffer, offset: resourceStride * index)
                
                for (resourceIndex, resource) in submesh.resources.enumerated() {
                    if let buffer = resource as? MTLBuffer {
                        encoder.setBuffer(buffer, offset: 0, index: resourceIndex)
                    } else if let texture = resource as? MTLTexture {
                        encoder.setTexture(texture, index: resourceIndex)
                    }
                }
            }
            
        }
        
        // Create skinned buffers (and persistent per-mesh uniform buffers)
        
        skinningPass.createBuffers(scene: scene)
        
        // For skinned meshes, point the normal resource at the skinned normal buffer
        // (the buffer pointer is stable; only its contents change each frame).
        var meshIndex = 0
        var skinnedIndex = 0
        for model in scene.models {
            for mesh in model.meshes {
                if mesh.hasSkinning {
                    for (submeshIndex, submesh) in mesh.submeshes.enumerated() {
                        let index = meshIndex * maxSubmeshes + submeshIndex
                        let offset = resourceStride * index
                        guard let encoder = device.makeArgumentEncoder(for: submesh.resources) else { continue }
                        encoder.setArgumentBuffer(resourcesBuffer, offset: offset)
                        encoder.setBuffer(skinningPass.skinnedVertexBuffers[skinnedIndex], offset: 0, index: 0)
                        encoder.setBuffer(skinningPass.skinnedPrevVertexBuffers[skinnedIndex], offset: 0, index: 1)
                        encoder.setBuffer(skinningPass.skinnedNormalBuffers[skinnedIndex], offset: 0, index: 2)
                        encoder.setBuffer(submesh.indexBuffer, offset: 0, index: 3)
                        encoder.setBuffer(submesh.materialBuffer, offset: 0, index: 4)
#if os(macOS)
                        resourcesBuffer.didModifyRange(offset..<(offset + resourceStride))
#endif
                    }
                    skinnedIndex += 1
                }
                meshIndex += 1
            }
        }
        
#if os(macOS)
        resourcesBuffer.didModifyRange(0..<resourcesBuffer.length)
#endif
        
        
        // Create scratch buffer for AS rebuilds
        // Size estimation: 32MB should be enough for dynamic updates, but for safety lets go slightly larger or use a heuristic
        scratchBuffer = device.makeBuffer(length: 1024 * 1024 * 64, options: .storageModePrivate)
        scratchBuffer?.label = "AS Scractch Buffer"
    }
    
    func createLegacyAccelerationStructures() {
        let primitiveAccelerationStructureDescriptors = scene.models.flatMap(\.meshes).map { mesh -> MTLPrimitiveAccelerationStructureDescriptor in
            let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = mesh.geometryDescriptors
            return descriptor
        }
        
        self.primitiveAccelerationStructures = legacyCommandQueue.buildCompactedAccelerationStructures(for: primitiveAccelerationStructureDescriptors)
        
        var instanceDescriptors = scene.models.flatMap(\.meshes).enumerated().map { index, mesh -> MTLIndirectAccelerationStructureInstanceDescriptor in
            var descriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
            descriptor.accelerationStructureID = primitiveAccelerationStructures[index].gpuResourceID
            descriptor.mask = 0xFF
            descriptor.options = []
            descriptor.transformationMatrix = packedFloat4x3(from: mesh.transform)
            return descriptor
        }
        
        self.instanceDescriptorBuffer = device.makeBuffer(bytes: &instanceDescriptors, length: MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride * scene.models.flatMap(\.meshes).count, options: CommonStorageMode.options)
        self.instanceDescriptorBuffer?.label = "Instance Descriptor Buffer"
        if let instanceDescriptorBuffer {
            previousInstanceDescriptorBuffer = device.makeBuffer(length: instanceDescriptorBuffer.length, options: CommonStorageMode.options)
            previousInstanceDescriptorBuffer?.label = "Previous Instance Descriptor Buffer"
            if let previousInstanceDescriptorBuffer {
                memcpy(previousInstanceDescriptorBuffer.contents(), instanceDescriptorBuffer.contents(), instanceDescriptorBuffer.length)
#if os(macOS)
                previousInstanceDescriptorBuffer.didModifyRange(0..<instanceDescriptorBuffer.length)
#endif
            }
        }
        
        let instanceAccelerationStructureDescriptor = MTLInstanceAccelerationStructureDescriptor()
        let instanceCount = scene.models.reduce(0) { result, model in
            return result + model.meshes.count
        }
        instanceAccelerationStructureDescriptor.instanceCount = instanceCount
        instanceAccelerationStructureDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
        instanceAccelerationStructureDescriptor.instanceDescriptorType = .indirect
        
        instancedAccelarationStructure = legacyCommandQueue.buildCompactedAccelerationStructure(with: instanceAccelerationStructureDescriptor)!
    }
    let initialSkinning = true
    func createMTL4AccelerationStructures() {
        // Ensure mesh buffers are resident before building primitive AS
        rebuildResidencySet()
        
        let allMeshes = scene.models.flatMap(\.meshes)
        
        if initialSkinning{
            // Initial skinning pass to populate skinned vertex buffers
            if let commandBuffer = device.makeCommandBuffer(),
               let allocator = device.makeCommandAllocator() {
                commandBuffer.label = "Initial Skinning"
                commandBuffer.beginCommandBuffer(allocator: allocator)
                if let residencySet = commandQueueResidencySet {
                    commandBuffer.useResidencySet(residencySet)
                }
                
                // Ensure models are updated to initial state (rest pose or time 0)
                for model in scene.models {
                    model.update(deltaTime: 0)
                }
                skinningPass.performSkinning(commandBuffer: commandBuffer, scene: scene)
                
                commandBuffer.endCommandBuffer()
                commandQueue.commit([commandBuffer])
                
                let event = device.makeSharedEvent()!
                event.signaledValue = 0
                commandQueue.signalEvent(event, value: 1)
                while !event.wait(untilSignaledValue: 1, timeoutMS: 1000) {}
            }
        }
        
        // Build compacted AS for static meshes
        var staticIndices: [Int] = []
        var skinnedIndices: [Int] = []
        
        for (index, mesh) in allMeshes.enumerated() {
            if mesh.hasSkinning {
                skinnedIndices.append(index)
            } else {
                staticIndices.append(index)
            }
        }
        
        // Build compacted AS for static meshes
        let staticDescriptors = staticIndices.map { index -> MTL4PrimitiveAccelerationStructureDescriptor in
            let descriptor = MTL4PrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = allMeshes[index].geometryDescriptorsMTL4
            return descriptor
        }
        
        let staticAS = staticDescriptors.isEmpty ? [] : commandQueue.buildCompactedAccelerationStructures(for: staticDescriptors, residencySet: commandQueueResidencySet)
        
        // Build UNCOMPACTED AS for skinned meshes (need full size for rebuild)
        // Use skinnedVertexBuffers which should now be populated by the initial skinning pass
        let skinnedDescriptors = skinnedIndices.enumerated().map { (offset, index) -> MTL4PrimitiveAccelerationStructureDescriptor in
            // Fallback to source mesh (T-Pose) if initial skinning is skipped
            if !initialSkinning {
                let descriptor = MTL4PrimitiveAccelerationStructureDescriptor()
                descriptor.geometryDescriptors = allMeshes[index].geometryDescriptorsMTL4
                return descriptor
            }
            return makePrimitiveDescriptor(mesh: allMeshes[index], vertexBuffer: skinningPass.skinnedVertexBuffers[offset])
        }
        
        let skinnedAS = skinnedDescriptors.isEmpty ? [] : commandQueue.buildAccelerationStructures(for: skinnedDescriptors, residencySet: commandQueueResidencySet)
        
        
        
        // Merge back in original order
        self.primitiveAccelerationStructures = []
        var staticCursor = 0
        var skinnedCursor = 0
        for (_, mesh) in allMeshes.enumerated() {
            if mesh.hasSkinning {
                primitiveAccelerationStructures.append(skinnedAS[skinnedCursor])
                skinnedCursor += 1
            } else {
                primitiveAccelerationStructures.append(staticAS[staticCursor])
                staticCursor += 1
            }
        }
        
        var instanceDescriptors = allMeshes.enumerated().map { index, mesh -> MTLIndirectAccelerationStructureInstanceDescriptor in
            var descriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
            descriptor.transformationMatrix = packedFloat4x3(from: mesh.transform)
            descriptor.mask = 0xFF
            descriptor.options = []
            descriptor.intersectionFunctionTableOffset = 0
            descriptor.userID = 0
            descriptor.accelerationStructureID = primitiveAccelerationStructures[index].gpuResourceID
            return descriptor
        }
        
        self.instanceDescriptorBuffer = device.makeBuffer(bytes: &instanceDescriptors,
                                                          length: MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride * instanceDescriptors.count,
                                                          options: CommonStorageMode.options)
        self.instanceDescriptorBuffer?.label = "Instance Descriptor Buffer"
        if let instanceDescriptorBuffer {
            previousInstanceDescriptorBuffer = device.makeBuffer(length: instanceDescriptorBuffer.length, options: CommonStorageMode.options)
            previousInstanceDescriptorBuffer?.label = "Previous Instance Descriptor Buffer"
            if let previousInstanceDescriptorBuffer {
                memcpy(previousInstanceDescriptorBuffer.contents(), instanceDescriptorBuffer.contents(), instanceDescriptorBuffer.length)
#if os(macOS)
                previousInstanceDescriptorBuffer.didModifyRange(0..<instanceDescriptorBuffer.length)
#endif
            }
        }
#if os(macOS)
        if let instanceDescriptorBuffer {
            instanceDescriptorBuffer.didModifyRange(0..<instanceDescriptorBuffer.length)
        }
#endif
        // Ensure Primitive AS and Instance Buffer are resident before building Instanced AS
        if let residencySet = commandQueueResidencySet {
            var newAllocations: [any MTLAllocation] = []
            for pas in primitiveAccelerationStructures {
                newAllocations.append(pas)
            }
            if let instanceDescriptorBuffer {
                newAllocations.append(instanceDescriptorBuffer)
            }
            residencySet.addAllocations(newAllocations)
            residencySet.commit()
        }
        
        let instanceAccelerationStructureDescriptor = MTL4InstanceAccelerationStructureDescriptor()
        let instanceCount = instanceDescriptors.count
        instanceAccelerationStructureDescriptor.instanceCount = instanceCount
        instanceAccelerationStructureDescriptor.instanceDescriptorType = .indirect
        if let instanceDescriptorBuffer {
            instanceAccelerationStructureDescriptor.instanceDescriptorBuffer = MTL4BufferRange(bufferAddress: instanceDescriptorBuffer.gpuAddress,
                                                                                               length: UInt64(instanceDescriptorBuffer.length))
        }
        
        // Build un-compacted TLAS when refit isn't supported so rebuilds have enough size.
        let instanceSizes = device.accelerationStructureSizes(descriptor: instanceAccelerationStructureDescriptor)
        if instanceSizes.refitScratchBufferSize == 0 {
            instancedAccelarationStructure = commandQueue.buildAccelerationStructures(for: [instanceAccelerationStructureDescriptor], residencySet: commandQueueResidencySet).first!
        } else {
            instancedAccelarationStructure = commandQueue.buildCompactedAccelerationStructure(with: instanceAccelerationStructureDescriptor, residencySet: commandQueueResidencySet)!
        }
    }
    
    func updateUniforms(size: CGSize) {
        if viewMode == .tps {
            if scene.models.indices.contains(playerModelIndex) {
                let player = scene.models[playerModelIndex]
                // Target is player position + some height (e.g. 1.0 for center of body)
                cameraTarget = player.position + SIMD3<Float>(0, 1.0, 0)
            }
        } else {
            // In World mode, we might want to default to (0,0,0) or keep existing.
            // The original logic just used 'scene.cameraTarget' which was initialized to (0,0,0).
            // But if we switched FROM tps, we might want to reset or stay.
            // Let's reset to scene.cameraTarget (0,0,0) if we want "original".
            // However, scene.cameraTarget is not updated by orbit. Orbit updates renderer.camera[Azimuth/etc].
            // cameraTarget is a property of Renderer now (copied from scene in init).
            // Let's just not force it in World mode, but ensure it's (0,0,0) if we want "World" to mean "Center".
            // For now, let's reset it to (0,0,0) only if we strictly want "World Original".
            // Since the user said "world(original one)", I'll assume they mean looking at origin.
            cameraTarget = SIMD3<Float>(0, 0, 0)
        }
        
        scene.updateUniforms(size: size,
                             target: cameraTarget,
                             azimuth: cameraAzimuth,
                             elevation: cameraElevation,
                             distance: cameraDistance,
                             fovDegrees: cameraFovDegrees)
        
        let pointer = uniformBuffer.contents().advanced(by: uniformBufferOffset)
        let uniforms = pointer.bindMemory(to: Uniforms.self, capacity: 1)
        
        uniforms.pointee.camera = scene.camera
        uniforms.pointee.previousCamera = previousCamera ?? scene.camera
        uniforms.pointee.lightCount = Int32(scene.lights.count)
        uniforms.pointee.width = Int32(size.width)
        uniforms.pointee.height = Int32(size.height)
        uniforms.pointee.blocksWide = ((uniforms.pointee.width) + 15) / 16
        uniforms.pointee.frameIndex = frameIndex
        uniforms.pointee.samplesPerPixel = samplesPerPixel
        uniforms.pointee.samplesPerPixel = samplesPerPixel
        uniforms.pointee.maxBounces = maxBounces
        uniforms.pointee.debugTextureMode = debugTextureMode
        uniforms.pointee.accumulationWeight = accumulationWeight
        uniforms.pointee.enableDenoiseGBuffer = useTemporalDenoiser ? 1 : 0
        uniforms.pointee.shadingMode = Int32(shadingMode.rawValue)
        uniforms.pointee.enableMotionAdaptiveAccumulation = useMotionAdaptiveAccumulation ? 1 : 0
        uniforms.pointee.motionAccumulationMinWeight = motionAccumulationMinWeight
        uniforms.pointee.motionAccumulationLowThresholdPixels = motionAccumulationLowThresholdPixels
        uniforms.pointee.motionAccumulationHighThresholdPixels = motionAccumulationHighThresholdPixels
        frameIndex += 1
        
        // Save current camera for next frame
        previousCamera = scene.camera
    }
    
    func clampedRenderScale() -> Float {
        return renderScale
    }
    
    func scaledSize(for outputSize: CGSize, scale: Float) -> CGSize {
        let width = max(1, Int(round(outputSize.width * CGFloat(scale))))
        let height = max(1, Int(round(outputSize.height * CGFloat(scale))))
        return CGSize(width: width, height: height)
    }
    
    func createTextures(outputSize: CGSize) {
        let outputWidth = Int(outputSize.width)
        let outputHeight = Int(outputSize.height)
        guard outputWidth > 0, outputHeight > 0 else {
            return
        }
        
        let desiredScale = clampedRenderScale()
        let inputSize = scaledSize(for: outputSize, scale: desiredScale)
        let colorFormat: MTLPixelFormat = .rgba32Float
        
        let inputWidth = Int(inputSize.width)
        let inputHeight = Int(inputSize.height)
        guard inputWidth > 0, inputHeight > 0 else {
            return
        }
        
        let inputUsage: MTLTextureUsage = [.shaderRead, .shaderWrite]
        
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = colorFormat
        descriptor.textureType = .type2D
        descriptor.width = inputWidth
        descriptor.height = inputHeight
        
        // Stored in private memory because only the GPU will read or write this texture.
        descriptor.storageMode = .private
        descriptor.usage = inputUsage
        
        accumulationTargets = [device.makeTexture(descriptor: descriptor)!, device.makeTexture(descriptor: descriptor)!]
        accumulationTargets[0].label = "Accumulation Texture 1"
        accumulationTargets[1].label = "Accumulation Texture 2"
        
        // Create a texture containing a random integer value for each pixel. the sample
        // uses these values to decorrelate pixels while drawing pseudorandom numbers from the
        // Halton sequence.
        descriptor.pixelFormat = .r32Uint
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        
        randomTexture = device.makeTexture(descriptor: descriptor)
        randomTexture.label = "Random Texture"
        
        // Initialize random values.
        let texelCount = randomTexture.width * randomTexture.height
        let numbers = Array<UInt32>.init(unsafeUninitializedCapacity: texelCount) { buffer, initializedCount in
            for i in 0..<texelCount {
                buffer[i] = arc4random() % (1024 * 1024)
            }
            initializedCount = texelCount
        }
        
        numbers.withUnsafeBufferPointer { bufferPointer in
            randomTexture.replace(
                region: .init(
                    origin: .init(x: 0, y: 0, z: 0),
                    size: .init(width: randomTexture.width, height: randomTexture.height, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: bufferPointer.baseAddress!,
                bytesPerRow: MemoryLayout<UInt32>.stride * randomTexture.width
            )
        }
        
        // Create depth and motion textures (always needed for shader)
        let depthDescriptor = MTLTextureDescriptor()
        depthDescriptor.pixelFormat = .r32Float
        depthDescriptor.textureType = .type2D
        depthDescriptor.width = inputWidth
        depthDescriptor.height = inputHeight
        depthDescriptor.storageMode = .private
        depthDescriptor.usage = [.shaderRead, .shaderWrite]
        depthTexture = device.makeTexture(descriptor: depthDescriptor)
        depthTexture?.label = "Depth Texture"
        
        let motionDescriptor = MTLTextureDescriptor()
        motionDescriptor.pixelFormat = .rg16Float
        motionDescriptor.textureType = .type2D
        motionDescriptor.width = inputWidth
        motionDescriptor.height = inputHeight
        motionDescriptor.storageMode = .private
        motionDescriptor.usage = [.shaderRead, .shaderWrite]
        motionTexture = device.makeTexture(descriptor: motionDescriptor)
        motionTexture?.label = "Motion Texture"
        
        let diffuseDescriptor = MTLTextureDescriptor()
        diffuseDescriptor.pixelFormat = .rgba16Float
        diffuseDescriptor.textureType = .type2D
        diffuseDescriptor.width = inputWidth
        diffuseDescriptor.height = inputHeight
        diffuseDescriptor.storageMode = .private
        diffuseDescriptor.usage = [.shaderRead, .shaderWrite]
        diffuseAlbedoTexture = device.makeTexture(descriptor: diffuseDescriptor)
        diffuseAlbedoTexture?.label = "Diffuse Albedo Texture"
        
        let specularDescriptor = MTLTextureDescriptor()
        specularDescriptor.pixelFormat = .rgba16Float
        specularDescriptor.textureType = .type2D
        specularDescriptor.width = inputWidth
        specularDescriptor.height = inputHeight
        specularDescriptor.storageMode = .private
        specularDescriptor.usage = [.shaderRead, .shaderWrite]
        specularAlbedoTexture = device.makeTexture(descriptor: specularDescriptor)
        specularAlbedoTexture?.label = "Specular Albedo Texture"
        
        let normalDescriptor = MTLTextureDescriptor()
        normalDescriptor.pixelFormat = .rgba16Float
        normalDescriptor.textureType = .type2D
        normalDescriptor.width = inputWidth
        normalDescriptor.height = inputHeight
        normalDescriptor.storageMode = .private
        normalDescriptor.usage = [.shaderRead, .shaderWrite]
        normalTexture = device.makeTexture(descriptor: normalDescriptor)
        normalTexture?.label = "Normal Texture"
        
        let roughnessDescriptor = MTLTextureDescriptor()
        roughnessDescriptor.pixelFormat = .r16Float
        roughnessDescriptor.textureType = .type2D
        roughnessDescriptor.width = inputWidth
        roughnessDescriptor.height = inputHeight
        roughnessDescriptor.storageMode = .private
        roughnessDescriptor.usage = [.shaderRead, .shaderWrite]
        roughnessTexture = device.makeTexture(descriptor: roughnessDescriptor)
        roughnessTexture?.label = "Roughness Texture"
        
        lastDrawableSize = outputSize
        framePresenter.createTextures(outputSize: outputSize, colorFormat: colorFormat, renderSize: inputSize, accumulationTargets: accumulationTargets)
        rebuildResidencySet()
    }
    
    func rebuildResidencySet() {
        let residencySetDesc = MTLResidencySetDescriptor()
        residencySetDesc.initialCapacity = 64  // Increased to accommodate all allocations
        guard let residencySet = try? device.makeResidencySet(descriptor: residencySetDesc) else {
            return
        }
        
        var allocations: [any MTLAllocation] = []
        allocations.append(uniformBuffer)
        allocations.append(resourcesBuffer)
        if let instanceDescriptorBuffer {
            allocations.append(instanceDescriptorBuffer)
        }
        if let previousInstanceDescriptorBuffer {
            allocations.append(previousInstanceDescriptorBuffer)
        }
        allocations.append(scene.lightBuffer)
        if let randomTexture {
            allocations.append(randomTexture)
        }
        if let depthTexture {
            allocations.append(depthTexture)
        }
        if let motionTexture {
            allocations.append(motionTexture)
        }
        if let diffuseAlbedoTexture {
            allocations.append(diffuseAlbedoTexture)
        }
        if let specularAlbedoTexture {
            allocations.append(specularAlbedoTexture)
        }
        if let normalTexture {
            allocations.append(normalTexture)
        }
        if let roughnessTexture {
            allocations.append(roughnessTexture)
        }
        for texture in accumulationTargets {
            allocations.append(texture)
        }
        if let mtl4Presenter = framePresenter as? MTL4UpScaleRenderer,
           let upscaledTexture = mtl4Presenter.upscaledTexture {
            allocations.append(upscaledTexture)
        }
        for model in scene.models {
            for mesh in model.meshes {
                allocations.append(mesh.vertexBuffer)
                allocations.append(mesh.normalBuffer)
                if let uvBuffer = mesh.uvBuffer {
                    allocations.append(uvBuffer)
                }
                if let jointIndexBuffer = mesh.jointIndexBuffer {
                    allocations.append(jointIndexBuffer)
                }
                if let jointWeightBuffer = mesh.jointWeightBuffer {
                    allocations.append(jointWeightBuffer)
                }
                for submesh in mesh.submeshes {
                    allocations.append(submesh.indexBuffer)
                    allocations.append(submesh.materialBuffer)
                    allocations.append(submesh.baseColorTexture)
                    allocations.append(submesh.normalMapTexture)
                    allocations.append(submesh.roughnessTexture)
                    allocations.append(submesh.metallicTexture)
                    allocations.append(submesh.aoTexture)
                    allocations.append(submesh.opacityTexture)
                    allocations.append(submesh.emissionTexture)
                }
            }
        }
        for accelerationStructure in primitiveAccelerationStructures {
            allocations.append(accelerationStructure)
        }
        if let instancedAccelarationStructure {
            allocations.append(instancedAccelarationStructure)
        }
        
        if let scratchBuffer {
            allocations.append(scratchBuffer)
        }
        skinningPass.collectAllocations(allocations: &allocations)
        
        residencySet.addAllocations(allocations)
        residencySet.commit()
        
        // Remove old residency set before adding new one
        if let oldResidencySet = commandQueueResidencySet {
            commandQueue.removeResidencySet(oldResidencySet)
        }
        
        commandQueue.addResidencySet(residencySet)
        commandQueueResidencySet = residencySet
    }
    
    func update(size: CGSize) {
        // isSceneDirty handled in updateSkinningAndBLAS now, or we can trigger it differently
        // But updateSkinningAndBLAS is likely called in draw().
        // Let's ensure frameIndex is reset if dirty here if needed, or leave it to updateSkinningAndBLAS.
        // Actually, updateSkinningAndBLAS is called later.
        // Just remove the explicit updateAccelerationStructures call.
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
        updateUniforms(size: size)
    }
    
    enum ViewPreset: Int {
        case free = 0
        case front
        case back
        case left
        case right
        case top
        case bottom
        case isometric
    }
    
    enum ViewMode {
        case world
        case tps
    }
    
    var viewMode: ViewMode = .world {
        didSet {
            frameIndex = 0
        }
    }
    
    var playerModelIndex: Int = 0
    
    // Skinning methods moved to SkinningPass.swift
    func updateInstanceDescriptors() {
        if let instanceDescriptorBuffer {
            if let previousInstanceDescriptorBuffer {
                memcpy(previousInstanceDescriptorBuffer.contents(), instanceDescriptorBuffer.contents(), instanceDescriptorBuffer.length)
#if os(macOS)
                previousInstanceDescriptorBuffer.didModifyRange(0..<instanceDescriptorBuffer.length)
#endif
            }
            let allMeshes = scene.models.flatMap(\.meshes)
            if canUseMTL4AccelerationStructures(device: device) {
                var instanceDescriptors = allMeshes.enumerated().map { index, mesh -> MTLIndirectAccelerationStructureInstanceDescriptor in
                    var descriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
                    descriptor.transformationMatrix = packedFloat4x3(from: mesh.transform)
                    descriptor.mask = 0xFF
                    descriptor.options = []
                    descriptor.intersectionFunctionTableOffset = 0
                    descriptor.userID = 0
                    descriptor.accelerationStructureID = primitiveAccelerationStructures[index].gpuResourceID
                    return descriptor
                }
                instanceDescriptorBuffer.contents().copyMemory(from: &instanceDescriptors, byteCount: instanceDescriptors.count * MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride)
            } else {
                var instanceDescriptors = allMeshes.enumerated().map { index, mesh -> MTLIndirectAccelerationStructureInstanceDescriptor in
                    var descriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
                    descriptor.accelerationStructureID = primitiveAccelerationStructures[index].gpuResourceID
                    descriptor.mask = 0xFF
                    descriptor.options = []
                    descriptor.transformationMatrix = packedFloat4x3(from: mesh.transform)
                    return descriptor
                }
                instanceDescriptorBuffer.contents().copyMemory(from: &instanceDescriptors, byteCount: instanceDescriptors.count * MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride)
            }
#if os(macOS)
            instanceDescriptorBuffer.didModifyRange(0..<instanceDescriptorBuffer.length)
#endif
        }
    }

    private func makeLegacyPrimitiveDescriptor(mesh: Mesh, vertexBuffer: MTLBuffer) -> MTLPrimitiveAccelerationStructureDescriptor {
        let descriptors = mesh.submeshes.map { submesh -> MTLAccelerationStructureTriangleGeometryDescriptor in
            let d = MTLAccelerationStructureTriangleGeometryDescriptor()
            let vb = mesh.mtkMesh.vertexBuffers[0]
            d.vertexBuffer = vertexBuffer
            d.vertexBufferOffset = (vertexBuffer === vb.buffer) ? vb.offset : 0
            d.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            let ib = submesh.mtkSubmesh.indexBuffer
            d.indexBuffer = ib.buffer
            d.indexBufferOffset = ib.offset
            d.indexType = submesh.mtkSubmesh.indexType
            d.triangleCount = submesh.mtkSubmesh.indexCount / 3
            return d
        }

        let primitiveDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
        primitiveDescriptor.geometryDescriptors = descriptors
        return primitiveDescriptor
    }
    func refitLegacyAccelerationStructures(computeEncoder: MTL4ComputeCommandEncoder) {
        let allMeshes = scene.models.flatMap(\.meshes)
        guard !allMeshes.isEmpty else { return }
        guard let instanceDescriptorBuffer else { return }
        
        func makePrimitiveDescriptor(mesh: Mesh, vertexBuffer: MTLBuffer) -> MTLPrimitiveAccelerationStructureDescriptor {
            let descriptors = mesh.submeshes.map { submesh -> MTLAccelerationStructureTriangleGeometryDescriptor in
                let d = MTLAccelerationStructureTriangleGeometryDescriptor()
                let vb = mesh.mtkMesh.vertexBuffers[0]
                d.vertexBuffer = vertexBuffer
                d.vertexBufferOffset = (vertexBuffer === vb.buffer) ? vb.offset : 0
                d.vertexStride = MemoryLayout<SIMD3<Float>>.stride
                let ib = submesh.mtkSubmesh.indexBuffer
                d.indexBuffer = ib.buffer
                d.indexBufferOffset = ib.offset
                d.indexType = submesh.mtkSubmesh.indexType
                d.triangleCount = submesh.mtkSubmesh.indexCount / 3
                return d
            }
            
            let primitiveDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
            primitiveDescriptor.geometryDescriptors = descriptors
            return primitiveDescriptor
        }
        
        let instanceDescriptor = MTLInstanceAccelerationStructureDescriptor()
        instanceDescriptor.instanceCount = allMeshes.count
        instanceDescriptor.instanceDescriptorType = .indirect
        instanceDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
        
        var requiredScratchSize = 0
        for mesh in allMeshes where mesh.hasSkinning {
            let primitiveDescriptor = makePrimitiveDescriptor(mesh: mesh, vertexBuffer: mesh.vertexBuffer)
            let sizes = device.accelerationStructureSizes(descriptor: primitiveDescriptor)
            requiredScratchSize = max(requiredScratchSize, max(sizes.refitScratchBufferSize, sizes.buildScratchBufferSize))
        }
        let instanceSizes = device.accelerationStructureSizes(descriptor: instanceDescriptor)
        requiredScratchSize = max(requiredScratchSize, max(instanceSizes.refitScratchBufferSize, instanceSizes.buildScratchBufferSize))
        
        let scratchSize = max(requiredScratchSize, 1)
        if scratchBuffer == nil || scratchBuffer!.length < scratchSize {
            scratchBuffer = device.makeBuffer(length: scratchSize, options: .storageModePrivate)
            scratchBuffer?.label = "AS Refit Scratch Buffer"
        }
        guard let scratchBuffer else { return }
        
        guard let commandBuffer = legacyCommandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else { return }
        encoder.label = "Legacy AS Refit"
        
        for (index, mesh) in allMeshes.enumerated() where mesh.hasSkinning {
            let primitiveDescriptor = makePrimitiveDescriptor(mesh: mesh, vertexBuffer: mesh.vertexBuffer)
            let sizes = device.accelerationStructureSizes(descriptor: primitiveDescriptor)
            let canRefit = sizes.refitScratchBufferSize > 0
            if canRefit {
                encoder.refit(sourceAccelerationStructure: primitiveAccelerationStructures[index],
                              descriptor: primitiveDescriptor,
                              destinationAccelerationStructure: nil,
                              scratchBuffer: scratchBuffer,
                              scratchBufferOffset: 0)
            } else {
                encoder.build(accelerationStructure: primitiveAccelerationStructures[index],
                              descriptor: primitiveDescriptor,
                              scratchBuffer: scratchBuffer,
                              scratchBufferOffset: 0)
            }
        }
        
        if let instancedAccelarationStructure {
            let sizes = device.accelerationStructureSizes(descriptor: instanceDescriptor)
            let canRefit = sizes.refitScratchBufferSize > 0
            if canRefit {
                encoder.refit(sourceAccelerationStructure: instancedAccelarationStructure,
                              descriptor: instanceDescriptor,
                              destinationAccelerationStructure: nil,
                              scratchBuffer: scratchBuffer,
                              scratchBufferOffset: 0)
            } else {
                encoder.build(accelerationStructure: instancedAccelarationStructure,
                              descriptor: instanceDescriptor,
                              scratchBuffer: scratchBuffer,
                              scratchBufferOffset: 0)
            }
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    func refitMTL4AccelerationStructures(computeEncoder: MTL4ComputeCommandEncoder) {
        // --- AS Refit Pass ---
        func bufferRange(_ buffer: MTLBuffer, offset: Int = 0, length: Int? = nil) -> MTL4BufferRange {
            let rangeLength = length ?? (buffer.length - offset)
            return MTL4BufferRange(bufferAddress: buffer.gpuAddress + UInt64(offset), length: UInt64(rangeLength))
        }
        
        // Ensure the scratch buffer is large enough for refit.
        let instanceRefitDescriptor: MTL4InstanceAccelerationStructureDescriptor? = {
            guard let instanceDescriptorBuffer else { return nil }
            let descriptor = MTL4InstanceAccelerationStructureDescriptor()
            descriptor.instanceCount = scene.models.flatMap(\.meshes).count
            descriptor.instanceDescriptorType = .indirect
            descriptor.instanceDescriptorBuffer = MTL4BufferRange(bufferAddress: instanceDescriptorBuffer.gpuAddress,
                                                                  length: UInt64(instanceDescriptorBuffer.length))
            return descriptor
        }()
        
        // Ensure the scratch buffer is large enough for refit.
        var requiredRefitScratchSize = 0
        var skinnedCursor = 0
        for model in scene.models {
            for mesh in model.meshes where mesh.hasSkinning {
                let destPos = skinningPass.skinnedVertexBuffers[skinnedCursor]
                let primitiveDescriptor = makePrimitiveDescriptor(mesh: mesh, vertexBuffer: destPos)
                let sizes = device.accelerationStructureSizes(descriptor: primitiveDescriptor)
                requiredRefitScratchSize = max(requiredRefitScratchSize, max(sizes.refitScratchBufferSize, sizes.buildScratchBufferSize))
                skinnedCursor += 1
            }
        }
        if let instanceRefitDescriptor {
            let sizes = device.accelerationStructureSizes(descriptor: instanceRefitDescriptor)
            requiredRefitScratchSize = max(requiredRefitScratchSize, max(sizes.refitScratchBufferSize, sizes.buildScratchBufferSize))
        }
        
        if requiredRefitScratchSize > 0, (scratchBuffer == nil || scratchBuffer!.length < requiredRefitScratchSize) {
            scratchBuffer = device.makeBuffer(length: requiredRefitScratchSize, options: .storageModePrivate)
            scratchBuffer?.label = "AS Refit Scratch Buffer"
            if let scratchBuffer, let residencySet = commandQueueResidencySet {
                residencySet.addAllocations([scratchBuffer])
                residencySet.commit()
            } else {
                rebuildResidencySet()
            }
        }
        
        // Use the same encoder for AS refit (no need for separate encoder)
        var flatIndex = 0
        skinnedCursor = 0
        
        for model in scene.models {
            for mesh in model.meshes {
                if mesh.hasSkinning {
                    let destPos = skinningPass.skinnedVertexBuffers[skinnedCursor]
                    let primitiveDescriptor = makePrimitiveDescriptor(mesh: mesh, vertexBuffer: destPos)
                    let sizes = device.accelerationStructureSizes(descriptor: primitiveDescriptor)
                    
                    let canRefit = sizes.refitScratchBufferSize > 0
                    let requiredScratchSize = canRefit ? sizes.refitScratchBufferSize : sizes.buildScratchBufferSize
                    let scratchRange: MTL4BufferRange
                    if requiredScratchSize == 0 {
                        scratchRange = MTL4BufferRange(bufferAddress: 0, length: 0)
                    } else if let scratchBuffer {
                        scratchRange = bufferRange(scratchBuffer, length: requiredScratchSize)
                    } else {
                        // No scratch buffer available even though one is required; skip refit safely.
                        flatIndex += 1
                        skinnedCursor += 1
                        continue
                    }
                    
                    let asStructure = primitiveAccelerationStructures[flatIndex]
                    if canRefit {
                        computeEncoder.refit(sourceAccelerationStructure: asStructure,
                                             descriptor: primitiveDescriptor,
                                             destinationAccelerationStructure: nil,
                                             scratchBuffer: scratchRange,
                                             options: [])
                    } else {
                        // Rebuild when refit isn't supported.
                        computeEncoder.build(destinationAccelerationStructure: asStructure,
                                             descriptor: primitiveDescriptor,
                                             scratchBuffer: scratchRange)
                    }
                    
                    skinnedCursor += 1
                }
                flatIndex += 1
            }
        }
        
        // Refit TLAS as well so its bounds reflect the updated BLAS.
        if let instanceRefitDescriptor, let instancedAccelarationStructure {
            let sizes = device.accelerationStructureSizes(descriptor: instanceRefitDescriptor)
            let canRefit = sizes.refitScratchBufferSize > 0
            let requiredScratchSize = canRefit ? sizes.refitScratchBufferSize : sizes.buildScratchBufferSize
            let scratchRange: MTL4BufferRange
            if requiredScratchSize == 0 {
                scratchRange = MTL4BufferRange(bufferAddress: 0, length: 0)
            } else if let scratchBuffer {
                scratchRange = bufferRange(scratchBuffer, length: requiredScratchSize)
            } else {
                scratchRange = MTL4BufferRange(bufferAddress: 0, length: 0)
            }
            
            if canRefit {
                computeEncoder.refit(sourceAccelerationStructure: instancedAccelarationStructure,
                                     descriptor: instanceRefitDescriptor,
                                     destinationAccelerationStructure: nil,
                                     scratchBuffer: scratchRange,
                                     options: [])
            } else {
                // Rebuild when refit isn't supported.
                computeEncoder.build(destinationAccelerationStructure: instancedAccelarationStructure,
                                     descriptor: instanceRefitDescriptor,
                                     scratchBuffer: scratchRange)
            }
        }
    }

    func updateSkinningAndBLASLegacy() {
        if !skinningPass.updateSceneTimeAndAnimation(scene: scene) { return }

        let allMeshes = scene.models.flatMap(\.meshes)
        guard !allMeshes.isEmpty else { return }
        guard primitiveAccelerationStructures.count == allMeshes.count else { return }

        let skinnedMeshes = allMeshes.filter { $0.hasSkinning }

        // Copy current skinned positions to previous buffers for motion vectors.
        if !skinnedMeshes.isEmpty && !skinningPass.skinnedPrevVertexBuffers.isEmpty {
            guard let copyCommandBuffer = legacyCommandQueue.makeCommandBuffer(),
                  let blitEncoder = copyCommandBuffer.makeBlitCommandEncoder()
            else { return }
            blitEncoder.label = "Skinning Prev Copy (Legacy)"
            let copyCount = min(skinningPass.skinnedVertexBuffers.count, skinningPass.skinnedPrevVertexBuffers.count)
            for i in 0..<copyCount {
                blitEncoder.copy(from: skinningPass.skinnedVertexBuffers[i],
                                 sourceOffset: 0,
                                 to: skinningPass.skinnedPrevVertexBuffers[i],
                                 destinationOffset: 0,
                                 size: skinningPass.skinnedVertexBuffers[i].length)
            }
            blitEncoder.endEncoding()
            copyCommandBuffer.commit()
            copyCommandBuffer.waitUntilCompleted()
        }

        // Skinning pass (legacy encoder)
        if !skinnedMeshes.isEmpty {
            guard let skinningCommandBuffer = legacyCommandQueue.makeCommandBuffer(),
                  let computeEncoder = skinningCommandBuffer.makeComputeCommandEncoder()
            else { return }
            computeEncoder.label = "Skinning (Legacy)"
            skinningPass.dispatchSkinningLegacy(computeEncoder: computeEncoder, scene: scene)
            computeEncoder.endEncoding()
            skinningCommandBuffer.commit()
            skinningCommandBuffer.waitUntilCompleted()
        }

        // Rebuild compacted BLAS for skinned meshes using skinned vertex buffers.
        if !skinnedMeshes.isEmpty {
            var skinnedIndices: [Int] = []
            skinnedIndices.reserveCapacity(skinnedMeshes.count)
            for (index, mesh) in allMeshes.enumerated() where mesh.hasSkinning {
                skinnedIndices.append(index)
            }
            guard skinningPass.skinnedVertexBuffers.count >= skinnedIndices.count else { return }
            let skinnedDescriptors = skinnedIndices.enumerated().map { (skinnedOffset, meshIndex) -> MTLPrimitiveAccelerationStructureDescriptor in
                let mesh = allMeshes[meshIndex]
                let vertexBuffer = skinningPass.skinnedVertexBuffers[skinnedOffset]
                return makeLegacyPrimitiveDescriptor(mesh: mesh, vertexBuffer: vertexBuffer)
            }
            let skinnedAS = legacyCommandQueue.buildCompactedAccelerationStructures(for: skinnedDescriptors)
            if skinnedAS.count == skinnedIndices.count {
                for (offset, meshIndex) in skinnedIndices.enumerated() {
                    primitiveAccelerationStructures[meshIndex] = skinnedAS[offset]
                }
            }
        }

        // Update instance descriptors after BLAS rebuilds to capture new AS IDs + transforms.
        updateInstanceDescriptors()

        // Rebuild TLAS using updated instance descriptors.
        if let instanceDescriptorBuffer {
            let instanceDescriptor = MTLInstanceAccelerationStructureDescriptor()
            instanceDescriptor.instanceCount = allMeshes.count
            instanceDescriptor.instanceDescriptorType = .indirect
            instanceDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
            if let tlas = legacyCommandQueue.buildCompactedAccelerationStructure(with: instanceDescriptor) {
                instancedAccelarationStructure = tlas
            }
        }
    }
    
    func updateSkinningAndBLAS(commandBuffer: MTL4CommandBuffer) {
        if !skinningPass.updateSceneTimeAndAnimation(scene: scene) { return }
        
        updateInstanceDescriptors()
        
        // 2. Dispatch Skinning Kernel and Refit BLAS in one encoder for proper synchronization
        let skinnedMeshes = scene.models.flatMap { $0.meshes }.filter { $0.hasSkinning }
        // guard !skinnedMeshes.isEmpty else { return } // REMOVED GUARD
        
        // Copy current skinned positions to previous buffers for motion vectors.
        if !skinnedMeshes.isEmpty && !skinningPass.skinnedPrevVertexBuffers.isEmpty {
            if let blitEncoder = commandBuffer.makeComputeCommandEncoder() {
                blitEncoder.label = "Skinning Prev Copy"
                let copyCount = min(skinningPass.skinnedVertexBuffers.count, skinningPass.skinnedPrevVertexBuffers.count)
                for i in 0..<copyCount {
                    blitEncoder.copy(sourceBuffer: skinningPass.skinnedVertexBuffers[i],
                                     sourceOffset: 0,
                                     destinationBuffer: skinningPass.skinnedPrevVertexBuffers[i],
                                     destinationOffset: 0,
                                     size: skinningPass.skinnedVertexBuffers[i].length)
                }
                blitEncoder.endEncoding()
            }
        }
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Skinning + AS Refit"
        
        // --- Skinning Pass (using shared helper) ---
        if !skinnedMeshes.isEmpty {
            skinningPass.dispatchSkinning(computeEncoder: computeEncoder, scene: scene)
        }
        
        // Barrier to ensure skinning dispatch writes are visible to AS refit
        // Using intra-pass barrier: wait for dispatch stage to complete before AS operations
        
        //        computeEncoder.barrier(afterEncoderStages: .dispatch,
        //                               beforeEncoderStages: .accelerationStructure,
        //                               visibilityOptions: .device)
        
        if canUseMTL4AccelerationStructures(device: device){
            refitMTL4AccelerationStructures(computeEncoder: computeEncoder)
        }else{
            // TODO: BugFix
            refitLegacyAccelerationStructures(computeEncoder: computeEncoder)
        }
        computeEncoder.endEncoding()
    }
    @MainActor
    func orbit(deltaX: Float, deltaY: Float) {
        if viewMode == .tps { return }
        let sensitivity: Float = 0.005
        cameraAzimuth += deltaX * sensitivity
        cameraElevation = clampElevation(cameraElevation + deltaY * sensitivity)
        frameIndex = 0
    }
    
    @MainActor
    func zoom(delta: Float) {
        let scale = max(0.1, 1.0 - delta)
        cameraDistance = max(minCameraDistance, min(maxCameraDistance, cameraDistance * scale))
        frameIndex = 0
    }
    
    @MainActor
    func applyViewPreset(_ preset: ViewPreset) {
        let isoElevation = asinf(1.0 / sqrtf(3.0))
        switch preset {
        case .free:
            return
        case .front:
            cameraAzimuth = 0.0
        case .back:
            cameraAzimuth = Float.pi
        case .left:
            cameraAzimuth = -Float.pi / 2.0
        case .right:
            cameraAzimuth = Float.pi / 2.0
        case .top:
            cameraElevation = cameraElevationLimit
        case .bottom:
            cameraElevation = -cameraElevationLimit
        case .isometric:
            cameraAzimuth = Float.pi / 4.0
            cameraElevation = isoElevation
        }
        cameraElevation = clampElevation(cameraElevation)
        frameIndex = 0
    }
    
    private func clampElevation(_ value: Float) -> Float {
        return max(-cameraElevationLimit, min(cameraElevationLimit, value))
    }
    
    private func makePrimitiveDescriptor(mesh: Mesh, vertexBuffer: MTLBuffer) -> MTL4PrimitiveAccelerationStructureDescriptor {
        let descriptors = mesh.submeshes.map { submesh -> MTL4AccelerationStructureTriangleGeometryDescriptor in
            let d = MTL4AccelerationStructureTriangleGeometryDescriptor()
            d.vertexBuffer = MTL4BufferRange(bufferAddress: vertexBuffer.gpuAddress, length: UInt64(vertexBuffer.length))
            d.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            d.vertexFormat = .float3
            
            let ib = submesh.mtkSubmesh.indexBuffer
            d.indexBuffer = MTL4BufferRange(bufferAddress: ib.buffer.gpuAddress + UInt64(ib.offset), length: UInt64(ib.length))
            d.indexType = submesh.mtkSubmesh.indexType
            d.triangleCount = submesh.mtkSubmesh.indexCount / 3
            return d
        }
        
        let primitiveDescriptor = MTL4PrimitiveAccelerationStructureDescriptor()
        primitiveDescriptor.geometryDescriptors = descriptors
        return primitiveDescriptor
    }
    
    // Convert a 4x4 transform matrix to MTLPackedFloat4x3 expected by instance descriptors
    private func packedFloat4x3(from m: simd_float4x4) -> MTLPackedFloat4x3 {
        var packed = MTLPackedFloat4x3()
        // MTLPackedFloat4x3 stores 3 rows of 4 floats each (row-major packing)
        packed.columns.0 = MTLPackedFloat3(m.columns.0.x, m.columns.0.y, m.columns.0.z)
        packed.columns.1 = MTLPackedFloat3(m.columns.1.x, m.columns.1.y, m.columns.1.z)
        packed.columns.2 = MTLPackedFloat3(m.columns.2.x, m.columns.2.y, m.columns.2.z)
        packed.columns.3 = MTLPackedFloat3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        return packed
    }
}

extension Renderer: MTKViewDelegate {
    func draw(in view: MTKView) {
        let previousValueToWaitFor = gpuFrameIndex - UInt64(maxFramesInFlight)
        if !endFrameEvent.wait(untilSignaledValue: previousValueToWaitFor, timeoutMS: 100){
            return
        }
        guard let drawable = view.currentDrawable else { return }
        let outputSize = view.drawableSize
        let outputWidth = Int(outputSize.width)
        let outputHeight = Int(outputSize.height)
        if outputWidth <= 0 || outputHeight <= 0 {
            return
        }
        if accumulationTargets.count < 2 || randomTexture == nil ||
            Int(lastDrawableSize.width) != outputWidth ||
            Int(lastDrawableSize.height) != outputHeight {
            createTextures(outputSize: outputSize)
            frameIndex = 0
            if accumulationTargets.count < 2 || randomTexture == nil {
                return
            }
        }
        let renderWidth = accumulationTargets[0].width
        let renderHeight = accumulationTargets[0].height
        let renderSize = CGSize(width: renderWidth, height: renderHeight)
        update(size: renderSize)
        
        let commandAllocator = commandAllocators[uniformBufferIndex]
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        
        let width = renderWidth
        let height = renderHeight
        
        // Update Skinning and BLAS
        if canUseMTL4AccelerationStructures(device: device){
            updateSkinningAndBLAS(commandBuffer: commandBuffer)
        } else {
            updateSkinningAndBLASLegacy()
        }
        
        // process rays in 16x16 tiles for better GPU utilization
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (height + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Raytracing Pass"
        computeEncoder.setComputePipelineState(raytracingPipelineState)
        
        let uniformAddress = uniformBuffer.gpuAddress + UInt64(uniformBufferOffset)
        computeTable.setAddress(uniformAddress, index: BufferIndex.uniforms.rawValue)
        computeTable.setAddress(resourcesBuffer.gpuAddress, index: BufferIndex.resources.rawValue)
        computeTable.setAddress(instanceDescriptorBuffer!.gpuAddress, index: BufferIndex.instanceDescriptors.rawValue)
        computeTable.setAddress(previousInstanceDescriptorBuffer!.gpuAddress, index: BufferIndex.previousInstanceDescriptors.rawValue)
        computeTable.setAddress(scene.lightBuffer.gpuAddress, index: BufferIndex.lights.rawValue)
        computeTable.setResource(instancedAccelarationStructure.gpuResourceID, bufferIndex: BufferIndex.accelerationStructure.rawValue)
        
        computeTable.setTexture(randomTexture.gpuResourceID, index: TextureIndex.random.rawValue)
        computeTable.setTexture(accumulationTargets[0].gpuResourceID, index: TextureIndex.accumulation.rawValue)
        computeTable.setTexture(accumulationTargets[1].gpuResourceID, index: TextureIndex.previousAccumulation.rawValue)
        
        if let depthTexture = depthTexture {
            computeTable.setTexture(depthTexture.gpuResourceID, index: TextureIndex.depth.rawValue)
        }
        if let motionTexture = motionTexture {
            computeTable.setTexture(motionTexture.gpuResourceID, index: TextureIndex.motion.rawValue)
        }
        if let diffuseAlbedoTexture = diffuseAlbedoTexture {
            computeTable.setTexture(diffuseAlbedoTexture.gpuResourceID, index: TextureIndex.diffuseAlbedo.rawValue)
        }
        if let specularAlbedoTexture = specularAlbedoTexture {
            computeTable.setTexture(specularAlbedoTexture.gpuResourceID, index: TextureIndex.specularAlbedo.rawValue)
        }
        if let normalTexture = normalTexture {
            computeTable.setTexture(normalTexture.gpuResourceID, index: TextureIndex.normal.rawValue)
        }
        if let roughnessTexture = roughnessTexture {
            computeTable.setTexture(roughnessTexture.gpuResourceID, index: TextureIndex.roughness.rawValue)
        }
        
        computeEncoder.setArgumentTable(computeTable)
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        let tmp = accumulationTargets[0]
        accumulationTargets[0] = accumulationTargets[1]
        accumulationTargets[1] = tmp
        
        if let residencySet = commandQueueResidencySet {
            commandBuffer.useResidencySet(residencySet)
        }
        
        framePresenter.draw(in: view, computeEvent: computeEvent, gpuFrameIndex: gpuFrameIndex, accumulationTargets: accumulationTargets, depthTexture: depthTexture, motionTexture: motionTexture, diffuseAlbedoTexture: diffuseAlbedoTexture, specularAlbedoTexture: specularAlbedoTexture, normalTexture: normalTexture, roughnessTexture: roughnessTexture, drawable: drawable, commandBuffer: commandBuffer)
        
        gpuFrameIndex += 1
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        if size.width <= 0 || size.height <= 0 {
            return
        }
        createTextures(outputSize: size)
        frameIndex = 0
    }
}

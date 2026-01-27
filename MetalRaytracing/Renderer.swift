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

class Renderer: NSObject {
    
    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let legacyCommandQueue: MTLCommandQueue // Add this for legacy AS building
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let library: MTLLibrary
    let computeTable: MTL4ArgumentTable
    let renderTable: MTL4ArgumentTable
    var commandQueueResidencySet: MTLResidencySet?
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
    
    // Max ray bounces (higher = more realistic lighting, slower)
    var maxBounces: Int32 = 2 {
        didSet {
            frameIndex = 0
        }
    }
    
    var uniformBuffer: MTLBuffer!
    var resourcesBuffer: MTLBuffer!
    var instanceDescriptorBuffer: MTLBuffer!
    
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
        computeTableDesc.maxBufferBindCount = 10
        computeTableDesc.maxTextureBindCount = 5
        self.computeTable = try! device.makeArgumentTable(descriptor: computeTableDesc)
        
        let renderTableDesc = MTL4ArgumentTableDescriptor()
        renderTableDesc.maxTextureBindCount = 1
        self.renderTable = try! device.makeArgumentTable(descriptor: renderTableDesc)
        
        self.endFrameEvent = device.makeSharedEvent()!
        self.computeEvent = device.makeSharedEvent()!
        self.gpuFrameIndex = UInt64(maxFramesInFlight)
        self.endFrameEvent.signaledValue = UInt64(maxFramesInFlight - 1)
        
        self.scene = DragonScene(size: size, device: device)
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
            self.framePresenter = LegacyFramePresenter(renderer: LecacyUpScaleRenderer(device: device, library: library, endFrameEvent: endFrameEvent)!, commandQueue: self.commandQueue)
        }
        
        if canUseMTL4AccelerationStructures(device: device){
            createMTL4AccelerationStructures()
        }else{
            createAccelerationStructures()
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
        
        print(raytracingPipelineState.threadExecutionWidth,
        raytracingPipelineState.maxTotalThreadsPerThreadgroup,
        raytracingPipelineState.staticThreadgroupMemoryLength)
    }

    func createBuffers() {
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride * maxFramesInFlight, options: .storageModeShared)!
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
        
#if os(macOS)
        resourcesBuffer.didModifyRange(0..<resourcesBuffer.length)
#endif
    }

    func createAccelerationStructures() {
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
            descriptor.transformationMatrix = .matrix4x4_drop_last_row(mesh.transform)
            return descriptor
        }
        
        self.instanceDescriptorBuffer = device.makeBuffer(bytes: &instanceDescriptors, length: MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride * scene.models.flatMap(\.meshes).count, options: CommonStorageMode.options)
        self.instanceDescriptorBuffer?.label = "Instance Descriptor Buffer"
        
        let instanceAccelerationStructureDescriptor = MTLInstanceAccelerationStructureDescriptor()
        let instanceCount = scene.models.reduce(0) { result, model in
            return result + model.meshes.count
        }
        instanceAccelerationStructureDescriptor.instanceCount = instanceCount
        instanceAccelerationStructureDescriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
        instanceAccelerationStructureDescriptor.instanceDescriptorType = .indirect
        
        instancedAccelarationStructure = legacyCommandQueue.buildCompactedAccelerationStructure(with: instanceAccelerationStructureDescriptor)!
    }
    func createMTL4AccelerationStructures() {
        // Ensure mesh buffers are resident before building primitive AS
        rebuildResidencySet()
        
        let primitiveAccelerationStructureDescriptors = scene.models.flatMap(\.meshes).map { mesh -> MTL4PrimitiveAccelerationStructureDescriptor in
            let descriptor = MTL4PrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = mesh.geometryDescriptorsMTL4
            return descriptor
        }

        self.primitiveAccelerationStructures = commandQueue.buildCompactedAccelerationStructures(for: primitiveAccelerationStructureDescriptors, residencySet: commandQueueResidencySet)

        var instanceDescriptors = scene.models.flatMap(\.meshes).enumerated().map { index, mesh -> MTLIndirectAccelerationStructureInstanceDescriptor in
            var descriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
            descriptor.transformationMatrix = .matrix4x4_drop_last_row(mesh.transform)
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

        instancedAccelarationStructure = commandQueue.buildCompactedAccelerationStructure(with: instanceAccelerationStructureDescriptor, residencySet: commandQueueResidencySet)!
    }
    
    func updateUniforms(size: CGSize) {
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
        uniforms.pointee.maxBounces = maxBounces
        frameIndex += 1
        
        // Save current camera for next frame
        previousCamera = scene.camera
    }
    
    private func clampedRenderScale() -> Float {
        return renderScale
    }

    private func scaledSize(for outputSize: CGSize, scale: Float) -> CGSize {
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

        lastDrawableSize = outputSize
        framePresenter.createTextures(outputSize: outputSize, colorFormat: colorFormat, renderSize: inputSize, accumulationTargets: accumulationTargets)
        rebuildResidencySet()
    }

    private func rebuildResidencySet() {
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
        for texture in accumulationTargets {
            allocations.append(texture)
        }
        for model in scene.models {
            for mesh in model.meshes {
                allocations.append(mesh.vertexBuffer)
                allocations.append(mesh.normalBuffer)
                for submesh in mesh.submeshes {
                    allocations.append(submesh.indexBuffer)
                    allocations.append(submesh.materialBuffer)
                }
            }
        }
        for accelerationStructure in primitiveAccelerationStructures {
            allocations.append(accelerationStructure)
        }
        if let instancedAccelarationStructure {
            allocations.append(instancedAccelarationStructure)
        }
        
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
    
    @MainActor
    func orbit(deltaX: Float, deltaY: Float) {
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
        computeTable.setAddress(instanceDescriptorBuffer.gpuAddress, index: BufferIndex.instanceDescriptors.rawValue)
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
        
        computeEncoder.setArgumentTable(computeTable)
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid: threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        let tmp = accumulationTargets[0]
        accumulationTargets[0] = accumulationTargets[1]
        accumulationTargets[1] = tmp

        if let residencySet = commandQueueResidencySet {
            commandBuffer.useResidencySet(residencySet)
        }
        
        framePresenter.draw(in: view, computeEvent: computeEvent, gpuFrameIndex: gpuFrameIndex, accumulationTargets: accumulationTargets, depthTexture: depthTexture, motionTexture: motionTexture, drawable: drawable, commandBuffer: commandBuffer)
        
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

class LecacyUpScaleRenderer {
    let device: MTLDevice
    let legacyCommandQueue: MTLCommandQueue
    let endFrameEvent: MTLSharedEvent
    let legacyRenderPipelineState: MTLRenderPipelineState
    
    var legacySpatialScaler: MTLFXSpatialScaler?
    var temporalScaler: MTLFXTemporalScaler?
    var upscaledTexture: MTLTexture?
    var useTemporalScaler: Bool = false
    var isMetalFXEnabled: Bool = true
    
    init?(device: MTLDevice, library: MTLLibrary, endFrameEvent: MTLSharedEvent) {
        self.device = device
        self.endFrameEvent = endFrameEvent
        
        guard let commandQueue = device.makeCommandQueue() else {
            return nil
        }
        self.legacyCommandQueue = commandQueue
        
        let legacyDescriptor = MTLRenderPipelineDescriptor()
        legacyDescriptor.vertexFunction = library.makeFunction(name: "vertexShader")
        legacyDescriptor.fragmentFunction = library.makeFunction(name: "fragmentShader")
        legacyDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        legacyDescriptor.label = "Legacy Render PipelineState"
        
        do {
            self.legacyRenderPipelineState = try device.makeRenderPipelineState(descriptor: legacyDescriptor)
        } catch {
            print("Failed to create legacy render pipeline state: \(error.localizedDescription)")
            return nil
        }
    }
    
    private func makeLegacySpatialScaler(inputWidth: Int,
                                         inputHeight: Int,
                                         outputWidth: Int,
                                         outputHeight: Int,
                                         colorFormat: MTLPixelFormat) -> MTLFXSpatialScaler? {
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else { return nil }
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.colorTextureFormat = colorFormat
        descriptor.outputTextureFormat = colorFormat
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorProcessingMode = .hdr
        return descriptor.makeSpatialScaler(device: device)
    }
    
    private func makeTemporalScaler(inputWidth: Int,
                                    inputHeight: Int,
                                    outputWidth: Int,
                                    outputHeight: Int,
                                    colorFormat: MTLPixelFormat) -> MTLFXTemporalScaler? {
        guard MTLFXTemporalScalerDescriptor.supportsDevice(device) else { return nil }
        let descriptor = MTLFXTemporalScalerDescriptor()
        descriptor.colorTextureFormat = colorFormat
        descriptor.depthTextureFormat = .r32Float
        descriptor.motionTextureFormat = .rg16Float
        descriptor.outputTextureFormat = colorFormat
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.isAutoExposureEnabled = false
        descriptor.isInputContentPropertiesEnabled = false
        return descriptor.makeTemporalScaler(device: device)
    }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture]) {
        let outputWidth = Int(outputSize.width)
        let outputHeight = Int(outputSize.height)
        let inputWidth = Int(renderSize.width)
        let inputHeight = Int(renderSize.height)
        
        guard outputWidth > 0, outputHeight > 0, inputWidth > 0, inputHeight > 0 else {
            return
        }
        
        let wantsUpscale = (inputWidth != outputWidth || inputHeight != outputHeight) && isMetalFXEnabled
        var spatialScaler: MTLFXSpatialScaler?
        var tempScaler: MTLFXTemporalScaler?
        let finalColorFormat = colorFormat  // Use the provided colorFormat directly
        
        if wantsUpscale {
            // Try temporal scaler first if enabled
            if useTemporalScaler {
                tempScaler = makeTemporalScaler(inputWidth: inputWidth,
                                               inputHeight: inputHeight,
                                               outputWidth: outputWidth,
                                               outputHeight: outputHeight,
                                               colorFormat: finalColorFormat)
            }
            
            // Fallback to spatial scaler if temporal is not available
            if tempScaler == nil {
                spatialScaler = makeLegacySpatialScaler(inputWidth: inputWidth,
                                                 inputHeight: inputHeight,
                                                 outputWidth: outputWidth,
                                                 outputHeight: outputHeight,
                                                 colorFormat: finalColorFormat)
            }
        }
        
        legacySpatialScaler = spatialScaler
        temporalScaler = tempScaler
        
        if wantsUpscale && (spatialScaler != nil || tempScaler != nil) {
            var outputUsage: MTLTextureUsage = [.shaderRead]
            if let scaler = legacySpatialScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            if let scaler = temporalScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            
            let upscaleDescriptor = MTLTextureDescriptor()
            upscaleDescriptor.pixelFormat = finalColorFormat
            upscaleDescriptor.textureType = .type2D
            upscaleDescriptor.width = outputWidth
            upscaleDescriptor.height = outputHeight
            upscaleDescriptor.storageMode = .private
            upscaleDescriptor.usage = outputUsage
            upscaledTexture = device.makeTexture(descriptor: upscaleDescriptor)
            upscaledTexture?.label = "Upscaled Texture"
        } else {
            upscaledTexture = nil
        }
    }
    
    func draw(in view: MTKView, computeEvent: MTLEvent, gpuFrameIndex: UInt64, accumulationTargets: [MTLTexture], depthTexture: MTLTexture?, motionTexture: MTLTexture?, drawable: CAMetalDrawable) {
        guard let legacyCommandBuffer = legacyCommandQueue.makeCommandBuffer() else {
            return
        }
        legacyCommandBuffer.encodeWaitForEvent(computeEvent, value: gpuFrameIndex)

        if let temporalScaler = temporalScaler,
           let upscaledTexture = upscaledTexture,
           let depthTexture = depthTexture,
           let motionTexture = motionTexture,
           !accumulationTargets.isEmpty {
            temporalScaler.colorTexture = accumulationTargets[0]
            temporalScaler.depthTexture = depthTexture
            temporalScaler.motionTexture = motionTexture
            temporalScaler.outputTexture = upscaledTexture
            temporalScaler.inputContentWidth = accumulationTargets[0].width
            temporalScaler.inputContentHeight = accumulationTargets[0].height
            temporalScaler.encode(commandBuffer: legacyCommandBuffer)
        } else if let legacySpatialScaler = legacySpatialScaler,
                  let upscaledTexture = upscaledTexture,
                  !accumulationTargets.isEmpty {
            legacySpatialScaler.colorTexture = accumulationTargets[0]
            legacySpatialScaler.inputContentWidth = accumulationTargets[0].width
            legacySpatialScaler.inputContentHeight = accumulationTargets[0].height
            legacySpatialScaler.outputTexture = upscaledTexture
            legacySpatialScaler.encode(commandBuffer: legacyCommandBuffer)
        }

        guard let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
        guard let renderEncoder = legacyCommandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        renderEncoder.setRenderPipelineState(legacyRenderPipelineState)
        
        // MARK: draw call
        let displayTexture = upscaledTexture ?? (accumulationTargets.isEmpty ? nil : accumulationTargets[0])
        if let displayTexture = displayTexture {
            renderEncoder.setFragmentTexture(displayTexture, index: 0)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }
        renderEncoder.endEncoding()

        legacyCommandBuffer.present(drawable)
        legacyCommandBuffer.encodeSignalEvent(endFrameEvent, value: gpuFrameIndex)
        legacyCommandBuffer.commit()
    }
    func buildCompactedAccelerationStructures(for descriptors: [MTLAccelerationStructureDescriptor],commandQueue:MTL4CommandQueue) -> [MTLAccelerationStructure] {
        return legacyCommandQueue.buildCompactedAccelerationStructures(for:descriptors)
    }
    func buildCompactedAccelerationStructure<T: MTLAccelerationStructureDescriptor>(with descriptor: T,commandQueue:MTL4CommandQueue) -> MTLAccelerationStructure? {
        return self.legacyCommandQueue.buildCompactedAccelerationStructure(with:descriptor)
    }
}
class MTL4UpScaleRenderer {
    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let endFrameEvent: MTLSharedEvent
    let renderPipelineState: MTLRenderPipelineState
    let commandBuffer: MTL4CommandBuffer
    let commandAllocator: MTL4CommandAllocator
    let renderTable: MTL4ArgumentTable
    
    var spatialScaler: MTL4FXSpatialScaler?
    var temporalScaler: MTL4FXTemporalScaler?
    var upscaledTexture: MTLTexture?
    var useTemporalScaler: Bool = false
    var isMetalFXEnabled: Bool = true
    let metal4Compiler: MTL4Compiler
    init?(device: MTLDevice, library: MTLLibrary, endFrameEvent: MTLSharedEvent, commandQueue: MTL4CommandQueue, metal4Compiler: MTL4Compiler) {
        self.device = device
        self.endFrameEvent = endFrameEvent
        self.commandQueue = commandQueue
        self.metal4Compiler = metal4Compiler
        guard let commandBuffer = device.makeCommandBuffer(),
              let commandAllocator = device.makeCommandAllocator() else {
            return nil
        }
        self.commandBuffer = commandBuffer
        self.commandAllocator = commandAllocator
        
        let renderTableDesc = MTL4ArgumentTableDescriptor()
        renderTableDesc.maxTextureBindCount = 1
        self.renderTable = try! device.makeArgumentTable(descriptor: renderTableDesc)
        
        let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        vertexFunctionDescriptor.library = library
        vertexFunctionDescriptor.name = "vertexShader"
        
        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        fragmentFunctionDescriptor.library = library
        fragmentFunctionDescriptor.name = "fragmentShader"
        
        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDescriptor
        pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDescriptor
        pipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        pipelineDescriptor.label = "Upscale Render PipelineState"
        
        do {
            self.renderPipelineState = try metal4Compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create upscale render pipeline state: \(error.localizedDescription)")
            return nil
        }
    }
    
    private func makeSpatialScaler(inputWidth: Int,
                                         inputHeight: Int,
                                         outputWidth: Int,
                                         outputHeight: Int,
                                         colorFormat: MTLPixelFormat) -> MTL4FXSpatialScaler? {
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else { return nil }
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.colorTextureFormat = colorFormat
        descriptor.outputTextureFormat = colorFormat
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorProcessingMode = .hdr
        return descriptor.makeSpatialScaler(device: device, compiler: metal4Compiler)
    }
    
    private func makeTemporalScaler(inputWidth: Int,
                                    inputHeight: Int,
                                    outputWidth: Int,
                                    outputHeight: Int,
                                    colorFormat: MTLPixelFormat) -> MTL4FXTemporalScaler? {
        guard MTLFXTemporalScalerDescriptor.supportsDevice(device) else { return nil }
        let descriptor = MTLFXTemporalScalerDescriptor()
        descriptor.colorTextureFormat = colorFormat
        descriptor.depthTextureFormat = .r32Float
        descriptor.motionTextureFormat = .rg16Float
        descriptor.outputTextureFormat = colorFormat
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.isAutoExposureEnabled = false
        descriptor.isInputContentPropertiesEnabled = false
        return descriptor.makeTemporalScaler(device: device, compiler: metal4Compiler)
    }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture]) {
        let outputWidth = Int(outputSize.width)
        let outputHeight = Int(outputSize.height)
        let inputWidth = Int(renderSize.width)
        let inputHeight = Int(renderSize.height)
        
        guard outputWidth > 0, outputHeight > 0, inputWidth > 0, inputHeight > 0 else {
            return
        }
        
        let wantsUpscale = (inputWidth != outputWidth || inputHeight != outputHeight) && isMetalFXEnabled
        var spaScaler: MTL4FXSpatialScaler?
        var tempScaler: MTL4FXTemporalScaler?
        let finalColorFormat = colorFormat  // Use the provided colorFormat directly
        
        if wantsUpscale {
            // Try temporal scaler first if enabled
            if useTemporalScaler {
                tempScaler = makeTemporalScaler(inputWidth: inputWidth,
                                               inputHeight: inputHeight,
                                               outputWidth: outputWidth,
                                               outputHeight: outputHeight,
                                                colorFormat: finalColorFormat)
            }
            
            // Fallback to spatial scaler if temporal is not available
            if tempScaler == nil {
                spaScaler = makeSpatialScaler(inputWidth: inputWidth,
                                                 inputHeight: inputHeight,
                                                 outputWidth: outputWidth,
                                                 outputHeight: outputHeight,
                                                 colorFormat: finalColorFormat)
            }
        }
        
        spatialScaler = spaScaler
        temporalScaler = tempScaler
        
        if wantsUpscale && (spatialScaler != nil || tempScaler != nil) {
            var outputUsage: MTLTextureUsage = [.shaderRead,.renderTarget,.shaderWrite]
            if let scaler = spatialScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            if let scaler = temporalScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            
            let upscaleDescriptor = MTLTextureDescriptor()
            upscaleDescriptor.pixelFormat = finalColorFormat
            upscaleDescriptor.textureType = .type2D
            upscaleDescriptor.width = outputWidth
            upscaleDescriptor.height = outputHeight
            upscaleDescriptor.storageMode = .private
            upscaleDescriptor.usage = outputUsage
            upscaledTexture = device.makeTexture(descriptor: upscaleDescriptor)
            upscaledTexture?.label = "Upscaled Texture"
        } else {
            upscaledTexture = nil
        }
    }
    
    func draw(in view: MTKView, computeEvent: MTLEvent, gpuFrameIndex: UInt64, accumulationTargets: [MTLTexture], depthTexture: MTLTexture?, motionTexture: MTLTexture?, drawable: CAMetalDrawable, commandBuffer passedCommandBuffer: MTL4CommandBuffer? = nil) {
        let commandBuffer: MTL4CommandBuffer
        if let passedCommandBuffer {
            commandBuffer = passedCommandBuffer
        } else {
            commandQueue.waitForEvent(computeEvent, value: gpuFrameIndex)
            commandAllocator.reset()
            self.commandBuffer.beginCommandBuffer(allocator: commandAllocator)
            commandBuffer = self.commandBuffer
        }

        if let temporalScaler = temporalScaler,
           let upscaledTexture = upscaledTexture,
           let depthTexture = depthTexture,
           let motionTexture = motionTexture,
           !accumulationTargets.isEmpty {
            temporalScaler.colorTexture = accumulationTargets[0]
            temporalScaler.depthTexture = depthTexture
            temporalScaler.motionTexture = motionTexture
            temporalScaler.outputTexture = upscaledTexture
            temporalScaler.inputContentWidth = accumulationTargets[0].width
            temporalScaler.inputContentHeight = accumulationTargets[0].height
            temporalScaler.fence = device.makeFence()
            temporalScaler.encode(commandBuffer: commandBuffer)
        } else if let spatialScaler = spatialScaler,
                  let upscaledTexture = upscaledTexture,
                  !accumulationTargets.isEmpty {
            spatialScaler.colorTexture = accumulationTargets[0]
            spatialScaler.inputContentWidth = accumulationTargets[0].width
            spatialScaler.inputContentHeight = accumulationTargets[0].height
            spatialScaler.outputTexture = upscaledTexture
            spatialScaler.fence = device.makeFence()
            spatialScaler.encode(commandBuffer: commandBuffer)
        }

        guard let renderPassDescriptor = view.currentMTL4RenderPassDescriptor else {
            commandBuffer.endCommandBuffer()
            commandQueue.commit([commandBuffer])
            commandQueue.signalEvent(endFrameEvent, value: gpuFrameIndex)
            return
        }
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            commandBuffer.endCommandBuffer()
            commandQueue.commit([commandBuffer])
            commandQueue.signalEvent(endFrameEvent, value: gpuFrameIndex)
            return
        }
        renderEncoder.setRenderPipelineState(renderPipelineState)
        
        // MARK: draw call
        let displayTexture = upscaledTexture ?? (accumulationTargets.isEmpty ? nil : accumulationTargets[0])
        if let displayTexture = displayTexture {
            renderTable.setTexture(displayTexture.gpuResourceID, index: 0)
            renderEncoder.setArgumentTable(renderTable, stages: [.fragment])
            renderEncoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
        }
        renderEncoder.endEncoding()

        commandBuffer.endCommandBuffer()
        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        commandQueue.signalEvent(endFrameEvent, value: gpuFrameIndex)
        drawable.present()
    }
    func buildCompactedAccelerationStructure<T: MTL4AccelerationStructureDescriptor>(with descriptor: T,commandQueue:MTL4CommandQueue) -> MTLAccelerationStructure? {
        return commandQueue.buildCompactedAccelerationStructure(with:descriptor)
    }
}


protocol FramePresenter {
    var useTemporalScaler: Bool { get set }
    var isMetalFXEnabled: Bool { get set }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture])
    
    func draw(in view: MTKView,
              computeEvent: MTLEvent,
              gpuFrameIndex: UInt64,
              accumulationTargets: [MTLTexture],
              depthTexture: MTLTexture?,
              motionTexture: MTLTexture?,
              drawable: CAMetalDrawable,
              commandBuffer: MTL4CommandBuffer)
}

class LegacyFramePresenter: FramePresenter {
    let renderer: LecacyUpScaleRenderer
    let commandQueue: MTL4CommandQueue
    
    var useTemporalScaler: Bool {
        get { renderer.useTemporalScaler }
        set { renderer.useTemporalScaler = newValue }
    }
    
    var isMetalFXEnabled: Bool {
        get { renderer.isMetalFXEnabled }
        set { renderer.isMetalFXEnabled = newValue }
    }
    
    init(renderer: LecacyUpScaleRenderer, commandQueue: MTL4CommandQueue) {
        self.renderer = renderer
        self.commandQueue = commandQueue
    }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture]) {
        renderer.createTextures(outputSize: outputSize, colorFormat: colorFormat, renderSize: renderSize, accumulationTargets: accumulationTargets)
    }
    
    func draw(in view: MTKView,
              computeEvent: MTLEvent,
              gpuFrameIndex: UInt64,
              accumulationTargets: [MTLTexture],
              depthTexture: MTLTexture?,
              motionTexture: MTLTexture?,
              drawable: CAMetalDrawable,
              commandBuffer: MTL4CommandBuffer) {
        
        // Emulate behavior: commit raytracing buffer, then signal, then legacy draw
        commandBuffer.endCommandBuffer()
        commandQueue.commit([commandBuffer])
        commandQueue.signalEvent(computeEvent, value: gpuFrameIndex)
        
        renderer.draw(in: view, computeEvent: computeEvent, gpuFrameIndex: gpuFrameIndex, accumulationTargets: accumulationTargets, depthTexture: depthTexture, motionTexture: motionTexture, drawable: drawable)
    }
}

extension MTL4UpScaleRenderer: FramePresenter {    
    func draw(in view: MTKView, computeEvent: MTLEvent, gpuFrameIndex: UInt64, accumulationTargets: [MTLTexture], depthTexture: MTLTexture?, motionTexture: MTLTexture?, drawable: CAMetalDrawable, commandBuffer: MTL4CommandBuffer) {
        self.draw(in: view, computeEvent: computeEvent, gpuFrameIndex: gpuFrameIndex, accumulationTargets: accumulationTargets, depthTexture: depthTexture, motionTexture: motionTexture, drawable: drawable, commandBuffer: commandBuffer as MTL4CommandBuffer?)
    }
}

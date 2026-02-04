import Metal
import MetalKit
import MetalFX
import simd

class LecacyUpScaleRenderer {
    let device: MTLDevice
    let legacyCommandQueue: MTLCommandQueue
    let endFrameEvent: MTLSharedEvent
    let legacyRenderPipelineState: MTLRenderPipelineState
    
    var legacySpatialScaler: MTLFXSpatialScaler?
    var temporalScaler: MTLFXTemporalScaler?
    @available(macOS 26.0, iOS 18.0, *)
    var temporalDenoisedScaler: MTLFXTemporalDenoisedScaler?
    var upscaledTexture: MTLTexture?
    var useTemporalScaler: Bool = false
    var useTemporalDenoiser: Bool = false
    var isMetalFXEnabled: Bool = true
    
    init?(device: MTLDevice, library: MTLLibrary, commandQueue: MTLCommandQueue ,endFrameEvent: MTLSharedEvent) {
        self.device = device
        self.endFrameEvent = endFrameEvent
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

    private func makeTemporalDenoisedScaler(inputWidth: Int,
                                            inputHeight: Int,
                                            outputWidth: Int,
                                            outputHeight: Int,
                                            colorFormat: MTLPixelFormat) -> MTLFXTemporalDenoisedScaler? {
        if #available(macOS 26.0, iOS 18.0, *) {
            guard MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(device) else { return nil }
            let descriptor = MTLFXTemporalDenoisedScalerDescriptor()
            descriptor.colorTextureFormat = colorFormat
            descriptor.depthTextureFormat = .r32Float
            descriptor.motionTextureFormat = .rg16Float
            descriptor.diffuseAlbedoTextureFormat = .rgba16Float
            descriptor.specularAlbedoTextureFormat = .rgba16Float
            descriptor.normalTextureFormat = .rgba16Float
            descriptor.roughnessTextureFormat = .r16Float
            descriptor.outputTextureFormat = colorFormat
            descriptor.inputWidth = inputWidth
            descriptor.inputHeight = inputHeight
            descriptor.outputWidth = outputWidth
            descriptor.outputHeight = outputHeight
            descriptor.isAutoExposureEnabled = false
            return descriptor.makeTemporalDenoisedScaler(device: device)
        }
        return nil
    }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture]) {
        let outputWidth = Int(outputSize.width)
        let outputHeight = Int(outputSize.height)
        let inputWidth = Int(renderSize.width)
        let inputHeight = Int(renderSize.height)
        
        guard outputWidth > 0, outputHeight > 0, inputWidth > 0, inputHeight > 0 else {
            return
        }
        
        let wantsScaler = isMetalFXEnabled && (useTemporalDenoiser || inputWidth != outputWidth || inputHeight != outputHeight)
        var spatialScaler: MTLFXSpatialScaler?
        var tempScaler: MTLFXTemporalScaler?
        var tempDenoiser: MTLFXTemporalDenoisedScaler?
        let finalColorFormat = colorFormat  // Use the provided colorFormat directly
        
        if wantsScaler {
            if useTemporalDenoiser {
                tempDenoiser = makeTemporalDenoisedScaler(inputWidth: inputWidth,
                                                          inputHeight: inputHeight,
                                                          outputWidth: outputWidth,
                                                          outputHeight: outputHeight,
                                                          colorFormat: finalColorFormat)
            }
            if tempDenoiser == nil && useTemporalScaler {
                tempScaler = makeTemporalScaler(inputWidth: inputWidth,
                                                inputHeight: inputHeight,
                                                outputWidth: outputWidth,
                                                outputHeight: outputHeight,
                                                colorFormat: finalColorFormat)
            }
            if tempDenoiser == nil && tempScaler == nil {
                spatialScaler = makeLegacySpatialScaler(inputWidth: inputWidth,
                                                        inputHeight: inputHeight,
                                                        outputWidth: outputWidth,
                                                        outputHeight: outputHeight,
                                                        colorFormat: finalColorFormat)
            }
        }
        
        legacySpatialScaler = spatialScaler
        temporalScaler = tempScaler
        temporalDenoisedScaler = tempDenoiser
        
        if wantsScaler && (spatialScaler != nil || tempScaler != nil || tempDenoiser != nil) {
            var outputUsage: MTLTextureUsage = [.shaderRead]
            if let scaler = legacySpatialScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            if let scaler = temporalScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            if let scaler = temporalDenoisedScaler {
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
    
    func draw(in view: MTKView, computeEvent: MTLEvent, gpuFrameIndex: UInt64, accumulationTargets: [MTLTexture], depthTexture: MTLTexture?, motionTexture: MTLTexture?, diffuseAlbedoTexture: MTLTexture?, specularAlbedoTexture: MTLTexture?, normalTexture: MTLTexture?, roughnessTexture: MTLTexture?, drawable: CAMetalDrawable) {
        guard let legacyCommandBuffer = legacyCommandQueue.makeCommandBuffer() else {
            return
        }
        legacyCommandBuffer.encodeWaitForEvent(computeEvent, value: gpuFrameIndex)
        
        if let temporalDenoisedScaler = temporalDenoisedScaler,
           let upscaledTexture = upscaledTexture,
           let depthTexture = depthTexture,
           let motionTexture = motionTexture,
           let diffuseAlbedoTexture = diffuseAlbedoTexture,
           let specularAlbedoTexture = specularAlbedoTexture,
           let normalTexture = normalTexture,
           let roughnessTexture = roughnessTexture,
           !accumulationTargets.isEmpty {
            temporalDenoisedScaler.colorTexture = accumulationTargets[0]
            temporalDenoisedScaler.depthTexture = depthTexture
            temporalDenoisedScaler.motionTexture = motionTexture
            temporalDenoisedScaler.diffuseAlbedoTexture = diffuseAlbedoTexture
            temporalDenoisedScaler.specularAlbedoTexture = specularAlbedoTexture
            temporalDenoisedScaler.normalTexture = normalTexture
            temporalDenoisedScaler.roughnessTexture = roughnessTexture
            temporalDenoisedScaler.outputTexture = upscaledTexture
//            temporalDenoisedScaler.inputContentWidth = accumulationTargets[0].width
//            temporalDenoisedScaler.inputContentHeight = accumulationTargets[0].height
            temporalDenoisedScaler.encode(commandBuffer: legacyCommandBuffer)
        } else if let temporalScaler = temporalScaler,
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
    @available(macOS 26.0, iOS 18.0, *)
    var temporalDenoisedScaler: MTL4FXTemporalDenoisedScaler?
    var upscaledTexture: MTLTexture?
    var useTemporalScaler: Bool = false
    var useTemporalDenoiser: Bool = false
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

    private func makeTemporalDenoisedScaler(inputWidth: Int,
                                            inputHeight: Int,
                                            outputWidth: Int,
                                            outputHeight: Int,
                                            colorFormat: MTLPixelFormat) -> MTL4FXTemporalDenoisedScaler? {
        if #available(macOS 26.0, iOS 18.0, *) {
            guard MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(device) else { return nil }
            let descriptor = MTLFXTemporalDenoisedScalerDescriptor()
            descriptor.colorTextureFormat = colorFormat
            descriptor.depthTextureFormat = .r32Float
            descriptor.motionTextureFormat = .rg16Float
            descriptor.diffuseAlbedoTextureFormat = .rgba16Float
            descriptor.specularAlbedoTextureFormat = .rgba16Float
            descriptor.normalTextureFormat = .rgba16Float
            descriptor.roughnessTextureFormat = .r16Float
            descriptor.outputTextureFormat = colorFormat
            descriptor.inputWidth = inputWidth
            descriptor.inputHeight = inputHeight
            descriptor.outputWidth = outputWidth
            descriptor.outputHeight = outputHeight
            descriptor.isAutoExposureEnabled = false
            return descriptor.makeTemporalDenoisedScaler(device: device, compiler: metal4Compiler)
        }
        return nil
    }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture]) {
        let outputWidth = Int(outputSize.width)
        let outputHeight = Int(outputSize.height)
        let inputWidth = Int(renderSize.width)
        let inputHeight = Int(renderSize.height)
        
        guard outputWidth > 0, outputHeight > 0, inputWidth > 0, inputHeight > 0 else {
            return
        }
        
        let wantsScaler = isMetalFXEnabled && (useTemporalDenoiser || inputWidth != outputWidth || inputHeight != outputHeight)
        var spaScaler: MTL4FXSpatialScaler?
        var tempScaler: MTL4FXTemporalScaler?
        var tempDenoiser: MTL4FXTemporalDenoisedScaler?
        let finalColorFormat = colorFormat  // Use the provided colorFormat directly
        
        if wantsScaler {
            if useTemporalDenoiser {
                tempDenoiser = makeTemporalDenoisedScaler(inputWidth: inputWidth,
                                                          inputHeight: inputHeight,
                                                          outputWidth: outputWidth,
                                                          outputHeight: outputHeight,
                                                          colorFormat: finalColorFormat)
            }
            if tempDenoiser == nil && useTemporalScaler {
                tempScaler = makeTemporalScaler(inputWidth: inputWidth,
                                                inputHeight: inputHeight,
                                                outputWidth: outputWidth,
                                                outputHeight: outputHeight,
                                                colorFormat: finalColorFormat)
            }
            if tempDenoiser == nil && tempScaler == nil {
                spaScaler = makeSpatialScaler(inputWidth: inputWidth,
                                              inputHeight: inputHeight,
                                              outputWidth: outputWidth,
                                              outputHeight: outputHeight,
                                              colorFormat: finalColorFormat)
            }
        }
        
        spatialScaler = spaScaler
        temporalScaler = tempScaler
        temporalDenoisedScaler = tempDenoiser
        
        if wantsScaler && (spatialScaler != nil || tempScaler != nil || tempDenoiser != nil) {
            var outputUsage: MTLTextureUsage = [.shaderRead]
            if let scaler = spatialScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            if let scaler = temporalScaler {
                outputUsage.formUnion(scaler.outputTextureUsage)
            }
            if let scaler = temporalDenoisedScaler {
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
    
    func draw(in view: MTKView, computeEvent: MTLEvent, gpuFrameIndex: UInt64, accumulationTargets: [MTLTexture], depthTexture: MTLTexture?, motionTexture: MTLTexture?, diffuseAlbedoTexture: MTLTexture?, specularAlbedoTexture: MTLTexture?, normalTexture: MTLTexture?, roughnessTexture: MTLTexture?, drawable: CAMetalDrawable, commandBuffer passedCommandBuffer: MTL4CommandBuffer? = nil) {
        let commandBuffer: MTL4CommandBuffer
        if let passedCommandBuffer {
            commandBuffer = passedCommandBuffer
        } else {
            commandQueue.waitForEvent(computeEvent, value: gpuFrameIndex)
            commandAllocator.reset()
            self.commandBuffer.beginCommandBuffer(allocator: commandAllocator)
            commandBuffer = self.commandBuffer
        }
        
        if let temporalDenoisedScaler = temporalDenoisedScaler,
           let upscaledTexture = upscaledTexture,
           let depthTexture = depthTexture,
           let motionTexture = motionTexture,
           let diffuseAlbedoTexture = diffuseAlbedoTexture,
           let specularAlbedoTexture = specularAlbedoTexture,
           let normalTexture = normalTexture,
           let roughnessTexture = roughnessTexture,
           !accumulationTargets.isEmpty {
            temporalDenoisedScaler.colorTexture = accumulationTargets[0]
            temporalDenoisedScaler.depthTexture = depthTexture
            temporalDenoisedScaler.motionTexture = motionTexture
            temporalDenoisedScaler.diffuseAlbedoTexture = diffuseAlbedoTexture
            temporalDenoisedScaler.specularAlbedoTexture = specularAlbedoTexture
            temporalDenoisedScaler.normalTexture = normalTexture
            temporalDenoisedScaler.roughnessTexture = roughnessTexture
            temporalDenoisedScaler.outputTexture = upscaledTexture
            // need fence otherwise "failed assertion `_outputTextureBarrierStages not set" raised
            temporalDenoisedScaler.fence = device.makeFence()
            temporalDenoisedScaler.encode(commandBuffer: commandBuffer)
        } else if let temporalScaler = temporalScaler,
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
            // need fence otherwise "failed assertion `_outputTextureBarrierStages not set" raised
            temporalScaler.fence = device.makeFence()
            temporalScaler.encode(commandBuffer: commandBuffer)
        } else if let spatialScaler = spatialScaler,
                  let upscaledTexture = upscaledTexture,
                  !accumulationTargets.isEmpty {
            spatialScaler.colorTexture = accumulationTargets[0]
            spatialScaler.inputContentWidth = accumulationTargets[0].width
            spatialScaler.inputContentHeight = accumulationTargets[0].height
            spatialScaler.outputTexture = upscaledTexture
            // need fence otherwise "failed assertion `_outputTextureBarrierStages not set" raised
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
    var useTemporalDenoiser: Bool { get set }
    var isMetalFXEnabled: Bool { get set }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture])
    
    func draw(in view: MTKView,
              computeEvent: MTLEvent,
              gpuFrameIndex: UInt64,
              accumulationTargets: [MTLTexture],
              depthTexture: MTLTexture?,
              motionTexture: MTLTexture?,
              diffuseAlbedoTexture: MTLTexture?,
              specularAlbedoTexture: MTLTexture?,
              normalTexture: MTLTexture?,
              roughnessTexture: MTLTexture?,
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
    
    var useTemporalDenoiser: Bool {
        get { renderer.useTemporalDenoiser }
        set { renderer.useTemporalDenoiser = newValue }
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
              diffuseAlbedoTexture: MTLTexture?,
              specularAlbedoTexture: MTLTexture?,
              normalTexture: MTLTexture?,
              roughnessTexture: MTLTexture?,
              drawable: CAMetalDrawable,
              commandBuffer: MTL4CommandBuffer) {
        
        // Emulate behavior: commit raytracing buffer, then signal, then legacy draw
        commandBuffer.endCommandBuffer()
        commandQueue.commit([commandBuffer])
        commandQueue.signalEvent(computeEvent, value: gpuFrameIndex)
        
        renderer.draw(in: view, computeEvent: computeEvent, gpuFrameIndex: gpuFrameIndex, accumulationTargets: accumulationTargets, depthTexture: depthTexture, motionTexture: motionTexture, diffuseAlbedoTexture: diffuseAlbedoTexture, specularAlbedoTexture: specularAlbedoTexture, normalTexture: normalTexture, roughnessTexture: roughnessTexture, drawable: drawable)
    }
}

extension MTL4UpScaleRenderer: FramePresenter {
    func draw(in view: MTKView, computeEvent: MTLEvent, gpuFrameIndex: UInt64, accumulationTargets: [MTLTexture], depthTexture: MTLTexture?, motionTexture: MTLTexture?, diffuseAlbedoTexture: MTLTexture?, specularAlbedoTexture: MTLTexture?, normalTexture: MTLTexture?, roughnessTexture: MTLTexture?, drawable: CAMetalDrawable, commandBuffer: MTL4CommandBuffer) {
        self.draw(in: view, computeEvent: computeEvent, gpuFrameIndex: gpuFrameIndex, accumulationTargets: accumulationTargets, depthTexture: depthTexture, motionTexture: motionTexture, diffuseAlbedoTexture: diffuseAlbedoTexture, specularAlbedoTexture: specularAlbedoTexture, normalTexture: normalTexture, roughnessTexture: roughnessTexture, drawable: drawable, commandBuffer: commandBuffer as MTL4CommandBuffer?)
    }
}

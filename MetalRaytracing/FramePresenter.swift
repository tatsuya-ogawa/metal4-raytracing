
protocol FramePresenter {
    var useTemporalScaler: Bool { get set }
    var useTemporalDenoiser: Bool { get set }
    var isMetalFXEnabled: Bool { get set }
    
    func createTextures(outputSize: CGSize, colorFormat: MTLPixelFormat, renderSize: CGSize, accumulationTargets: [MTLTexture])
    
    func draw(in view: MTKView,
              commandBuffer: MTL4CommandBuffer,
              commandQueue: MTLCommandQueue?,
              endFrameEvent: MTLSharedEvent?,
              computeEvent: MTLSharedEvent,
              gpuFrameIndex: UInt64,
              accumulationTargets: [MTLTexture],
              depthTexture: MTLTexture?,
              motionTexture: MTLTexture?,
              diffuseAlbedoTexture: MTLTexture?,
              specularAlbedoTexture: MTLTexture?,
              normalTexture: MTLTexture?,
              roughnessTexture: MTLTexture?,
              drawable: CAMetalDrawable)
}

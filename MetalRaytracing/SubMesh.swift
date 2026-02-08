//
//  SubMesh.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

let materialTextureBaseColor: UInt32 = 1 << 0
let materialTextureNormal: UInt32 = 1 << 1
let materialTextureRoughness: UInt32 = 1 << 2
let materialTextureMetallic: UInt32 = 1 << 3
let materialTextureAO: UInt32 = 1 << 4
let materialTextureEmission: UInt32 = 1 << 5
let materialTextureOpacity: UInt32 = 1 << 6

class Submesh {
    let mtkSubmesh: MTKSubmesh
    var material: Material
    
    let positionBuffer: MTLBuffer
    let previousPositionBuffer: MTLBuffer
    let normalBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let materialBuffer: MTLBuffer
    let uvBuffer: MTLBuffer?
    let baseColorTexture: MTLTexture
    let normalMapTexture: MTLTexture
    var roughnessTexture: MTLTexture
    var metallicTexture: MTLTexture
    var aoTexture: MTLTexture
    var opacityTexture: MTLTexture
    var emissionTexture: MTLTexture
    
    let mask: Int32
    
    var resources: [MTLResource] {
        var res: [MTLResource] = [positionBuffer, previousPositionBuffer, normalBuffer, indexBuffer, materialBuffer]
        if let uvBuffer = uvBuffer {
            res.append(uvBuffer)
        } else {
            // Fill the slot with a dummy buffer (reuse normalBuffer) to maintain index alignment for shader
            res.append(normalBuffer)
        }
        res.append(baseColorTexture)
        res.append(normalMapTexture)
        res.append(roughnessTexture)
        res.append(metallicTexture)
        res.append(aoTexture)
        res.append(opacityTexture)
        res.append(emissionTexture)
        return res
    }
    
    init(modelName: String, mdlSubmesh: MDLSubmesh, mtkSubmesh: MTKSubmesh, positionBuffer: MTLBuffer, normalBuffer: MTLBuffer, uvBuffer: MTLBuffer?, mask: Int32, on device: MTLDevice) {
        self.mtkSubmesh = mtkSubmesh
        self.material = Material(material: mdlSubmesh.material)
        self.positionBuffer = positionBuffer
        self.previousPositionBuffer = positionBuffer
        self.normalBuffer = normalBuffer
        self.uvBuffer = uvBuffer
        self.positionBuffer.label = "\(modelName) Positions"
        self.normalBuffer.label = "\(modelName) Normals"
        if let uvBuffer = uvBuffer {
            uvBuffer.label = "\(modelName) UVs"
        }
        
        // Load Texture
        let textureLoader = MTKTextureLoader(device: device)
        var texture: MTLTexture? = nil
        var normalTexture: MTLTexture? = nil
        var roughnessTexture: MTLTexture? = nil
        var metallicTexture: MTLTexture? = nil
        var aoTexture: MTLTexture? = nil
        var opacityTexture: MTLTexture? = nil
        var emissionTexture: MTLTexture? = nil
        if let material = mdlSubmesh.material {
            let colorOptions: [MTKTextureLoader.Option : Any] = [
                .generateMipmaps: true,
                .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                .textureStorageMode: MTLStorageMode.private.rawValue,
                .SRGB: true
            ]
            let normalOptions: [MTKTextureLoader.Option : Any] = [
                .generateMipmaps: true,
                .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                .textureStorageMode: MTLStorageMode.private.rawValue,
                .SRGB: false
            ]
            let linearOptions: [MTKTextureLoader.Option : Any] = [
                .generateMipmaps: true,
                .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                .textureStorageMode: MTLStorageMode.private.rawValue,
                .SRGB: false
            ]

            func loadTexture(from property: MDLMaterialProperty, options: [MTKTextureLoader.Option : Any]) -> MTLTexture? {
                if property.type == .string || property.type == .URL,
                   let source = property.urlValue ?? (property.stringValue != nil ? URL(string: property.stringValue!) : nil) {
                    do {
                        return try textureLoader.newTexture(URL: source, options: options)
                    } catch {
                        print("Failed to load texture for \(modelName): \(error)")
                        // Fallback attempt: check bundle resources if the URL seems relative
                        let filename = source.lastPathComponent
                        if let bundleURL = Bundle.main.url(forResource: "AssetResources/\(filename)", withExtension: nil) {
                            return try? textureLoader.newTexture(URL: bundleURL, options: options)
                        }
                    }
                } else if property.type == .texture,
                          let sampler = property.textureSamplerValue,
                          let mdlTexture = sampler.texture {
                    return try? textureLoader.newTexture(texture: mdlTexture, options: options)
                }
                return nil
            }
            
            if let baseColorProperty = material.property(with: .baseColor) {
                if let loaded = loadTexture(from: baseColorProperty, options: colorOptions) {
                    texture = loaded
                    self.material.textureFlags |= materialTextureBaseColor
                    self.material.baseColor = SIMD3<Float>(repeating: 1.0)
                }
            }
            
            if let normalProperty = material.property(with: .tangentSpaceNormal), normalProperty.type != .none {
                if let loaded = loadTexture(from: normalProperty, options: normalOptions) {
                    normalTexture = loaded
                    self.material.textureFlags |= materialTextureNormal
                }
            } else if let normalProperty = material.property(with: .objectSpaceNormal), normalProperty.type != .none {
                if let loaded = loadTexture(from: normalProperty, options: normalOptions) {
                    normalTexture = loaded
                    self.material.textureFlags |= materialTextureNormal
                }
            }
            if let roughnessProperty = material.property(with: .roughness) {
                if let loaded = loadTexture(from: roughnessProperty, options: linearOptions) {
                    roughnessTexture = loaded
                    self.material.textureFlags |= materialTextureRoughness
                }
            }
            
            if let metallicProperty = material.property(with: .metallic) {
                 if let loaded = loadTexture(from: metallicProperty, options: linearOptions) {
                     metallicTexture = loaded
                     self.material.textureFlags |= materialTextureMetallic
                 }
             }
            
#if ENABLE_AO
            if let aoProperty = material.property(with: .ambientOcclusion) {
                if let loaded = loadTexture(from: aoProperty, options: linearOptions) {
                    aoTexture = loaded
                    self.material.textureFlags |= materialTextureAO
                }
            }
#endif
            
            if let emissionProperty = material.property(with: .emission) {
                 if let loaded = loadTexture(from: emissionProperty, options: colorOptions) {
                     emissionTexture = loaded
                     self.material.textureFlags |= materialTextureEmission
                 }
             }

            if let opacityProperty = material.property(with: .opacity) {
                if let loaded = loadTexture(from: opacityProperty, options: linearOptions) {
                    opacityTexture = loaded
                    self.material.textureFlags |= materialTextureOpacity
                }
            }
        }
        
        // Fallback to white texture if no texture found or loading failed
        if texture == nil {
             let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
             descriptor.usage = .shaderRead
             if let whiteTex = device.makeTexture(descriptor: descriptor) {
                 var white: UInt32 = 0xFFFFFFFF
                 whiteTex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &white, bytesPerRow: 4)
                 texture = whiteTex
             } else {
                 fatalError("Failed to create dummy white texture")
             }
        }
        
        self.baseColorTexture = texture!

        // Fallback to neutral normal map if missing
        if normalTexture == nil {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
            descriptor.usage = .shaderRead
            if let neutralTex = device.makeTexture(descriptor: descriptor) {
                var neutral: UInt32 = 0xFFFF8080 // (0.5, 0.5, 1.0, 1.0) in RGBA8
                neutralTex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &neutral, bytesPerRow: 4)
                normalTexture = neutralTex
            } else {
                fatalError("Failed to create dummy normal texture")
            }
        }
        
        self.normalMapTexture = normalTexture!
        
        // Fallback for others (Roughness = 1.0 (White), Metallic = 0.0 (Black), AO = 1.0 (White), Opacity = 1.0 (White), Emission = 0.0 (Black))
        if roughnessTexture == nil || aoTexture == nil {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
            descriptor.usage = .shaderRead
            if let whiteTex = device.makeTexture(descriptor: descriptor) {
                var white: UInt32 = 0xFFFFFFFF
                whiteTex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &white, bytesPerRow: 4)
                if roughnessTexture == nil { roughnessTexture = whiteTex }
                if aoTexture == nil { aoTexture = whiteTex }
            }
        }
        self.roughnessTexture = roughnessTexture!
        self.aoTexture = aoTexture!
        if opacityTexture == nil {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
            descriptor.usage = .shaderRead
            if let whiteTex = device.makeTexture(descriptor: descriptor) {
                var white: UInt32 = 0xFFFFFFFF
                whiteTex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &white, bytesPerRow: 4)
                opacityTexture = whiteTex
            }
        }
        self.opacityTexture = opacityTexture!
        
        if metallicTexture == nil || emissionTexture == nil {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
            descriptor.usage = .shaderRead
            if let blackTex = device.makeTexture(descriptor: descriptor) {
                var black: UInt32 = 0x000000FF // Alpha 1.0, RGB 0.0
                blackTex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &black, bytesPerRow: 4)
                if metallicTexture == nil { metallicTexture = blackTex }
                if emissionTexture == nil { emissionTexture = blackTex }
            }
        }
        self.metallicTexture = metallicTexture!
        self.emissionTexture = emissionTexture!
        
        // Fix: Convert 16-bit indices to 32-bit to match Shader's `int* indices`
        let originalIndexBuffer = mtkSubmesh.indexBuffer.buffer
        let indexCount = mtkSubmesh.indexCount
        
        if mtkSubmesh.indexType == .uint16 {
            let indexLength = indexCount * MemoryLayout<UInt32>.stride
            guard let newIndexBuffer = device.makeBuffer(length: indexLength, options: CommonStorageMode.options) else {
                fatalError("Failed to allocate 32-bit index buffer")
            }
            newIndexBuffer.label = "\(modelName) Indices (Converted 32-bit)"
            
            let sourcePtr = originalIndexBuffer.contents().advanced(by: mtkSubmesh.indexBuffer.offset).bindMemory(to: UInt16.self, capacity: indexCount)
            let destPtr = newIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexCount)
            
            for i in 0..<indexCount {
                destPtr[i] = UInt32(sourcePtr[i])
            }
            
            self.indexBuffer = newIndexBuffer
        } else {
            self.indexBuffer = originalIndexBuffer
            self.indexBuffer.label = "\(modelName) Indices"
        }
        
        self.materialBuffer = device.makeBuffer(bytes: &self.material, length: MemoryLayout<Material>.stride, options: CommonStorageMode.options)!
        self.materialBuffer.label = "\(modelName) Material"
        self.mask = mask
    }

    func applyMaterialOverride(_ materialOverride: ModelMaterialOverride) {
        if let baseColor = materialOverride.baseColor {
            material.baseColor = baseColor
        }
        if let refractionIndex = materialOverride.refractionIndex {
            material.refractionIndex = max(refractionIndex, 1.0)
        }
        if let opacity = materialOverride.opacity {
            material.opacity = min(max(opacity, 0.0), 1.0)
        }

        var updatedMaterial = material
        materialBuffer.contents().copyMemory(from: &updatedMaterial, byteCount: MemoryLayout<Material>.stride)
#if os(macOS)
        materialBuffer.didModifyRange(0..<materialBuffer.length)
#endif
    }
}

private extension Material {
    init(material: MDLMaterial?) {
        self.init()
        self.textureFlags = 0
        self.refractionIndex = 1.0
        self.opacity = 1.0
        if let baseColor = material?.property(with: .baseColor), baseColor.type == .float3 {
            self.baseColor = baseColor.float3Value
        }else if let baseColor = material?.property(with: .baseColor), baseColor.type == .texture {
            // fallback
            self.baseColor = SIMD3<Float>(repeating:1.0)
        }
        if let emission = material?.property(with: .emission), emission.type == .float3 {
            self.emission = emission.float3Value
        }
        if let specular = material?.property(with: .specular), specular.type == .float3 {
            self.specular = specular.float3Value
        }
        if let specularExponent = material?.property(with: .specularExponent), specularExponent.type == .float3 {
            self.specularExponent = specularExponent.floatValue
        }
        if let refractionIndex = material?.property(with: .materialIndexOfRefraction), refractionIndex.type == .float {
            self.refractionIndex = refractionIndex.floatValue
        }
        if let opacity = material?.property(with: .opacity) {
            if opacity.type == .float {
                self.opacity = opacity.floatValue
            } else if opacity.type == .float3 {
                self.opacity = opacity.float3Value.x
            }
            self.opacity = min(max(self.opacity, 0.0), 1.0)
        }
    }
}

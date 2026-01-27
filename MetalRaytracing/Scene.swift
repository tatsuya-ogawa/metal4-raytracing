//
//  Scene.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

class Scene {
    var models: [Model]
    
    var camera: Camera
    var cameraTarget: SIMD3<Float>
    var cameraAzimuth: Float
    var cameraElevation: Float
    var cameraDistance: Float
    var cameraFovDegrees: Float

    var lights: [Light]
    var lightBuffer: MTLBuffer
    
    init(size: CGSize, device: MTLDevice) {
        self.camera = Scene.setupCamera(size: size)
        self.cameraTarget = SIMD3<Float>(0.0, 0.0, 0.0)
        let offset = camera.position - cameraTarget
        self.cameraDistance = max(0.001, simd_length(offset))
        self.cameraAzimuth = atan2(offset.x, offset.z)
        self.cameraElevation = asin(offset.y / self.cameraDistance)
        self.cameraFovDegrees = 45.0
        self.models = []
        let light1 = Scene.setupLight()
        var light2 = Scene.setupLight()
        light2.position = [2.0, 1.98, 3.0]
        light2.forward = [0, -0.5, 0.0]
        light2.right = [0.1, 0.0, 0.0]
        light2.up = [0.0, 0.0, 0.1]
        let light3 = Light.spotLight(position: [2, 1, 4], direction: [-1.5, -0.5, -1.5], coneAngle: 25 / 180 * .pi, color: [4,4,4])
        //let sunLight = Light.sunLight(direction: [-1, -2, 0], color: [1,1,1])
        //let pointLight = Light.pointLight(position: [1,1,1], color: [1, 1, 1])
        self.lights = [light1, light3]
        
        self.lightBuffer = device.makeBuffer(bytes: &lights, length: MemoryLayout<Light>.stride * lights.count, options: CommonStorageMode.options)!
        self.lightBuffer.label = "Lights Buffer"
    }
    
    func updateUniforms(size: CGSize,
                        target: SIMD3<Float>,
                        azimuth: Float,
                        elevation: Float,
                        distance: Float,
                        fovDegrees: Float) {
        camera = Self.makeOrbitCamera(size: size,
                                      target: target,
                                      azimuth: azimuth,
                                      elevation: elevation,
                                      distance: distance,
                                      fovDegrees: fovDegrees)
    }
    
    static func setupCamera(size: CGSize) -> Camera {
        let target = SIMD3<Float>(0.0, 0.0, 0.0)
        let defaultPosition = SIMD3<Float>(0.0, 1.0, 5.38)
        let offset = defaultPosition - target
        let distance = max(0.001, simd_length(offset))
        let azimuth = atan2(offset.x, offset.z)
        let elevation = asin(offset.y / distance)
        return makeOrbitCamera(size: size,
                               target: target,
                               azimuth: azimuth,
                               elevation: elevation,
                               distance: distance,
                               fovDegrees: 45.0)
    }

    static func makeOrbitCamera(size: CGSize,
                                target: SIMD3<Float>,
                                azimuth: Float,
                                elevation: Float,
                                distance: Float,
                                fovDegrees: Float) -> Camera {
        let safeDistance = max(0.001, distance)
        let limit = (Float.pi / 2.0) - 0.001
        let clampedElevation = max(-limit, min(limit, elevation))
        
        let x = safeDistance * cos(clampedElevation) * sin(azimuth)
        let y = safeDistance * sin(clampedElevation)
        let z = safeDistance * cos(clampedElevation) * cos(azimuth)
        let position = target + SIMD3<Float>(x, y, z)
        
        let forward = normalize(target - position)
        let worldUp = SIMD3<Float>(0.0, 1.0, 0.0)
        var right = normalize(cross(forward, worldUp))
        if simd_length(right) < 0.0001 {
            right = SIMD3<Float>(1.0, 0.0, 0.0)
        }
        let up = normalize(cross(right, forward))
        
        let fieldOfView = fovDegrees * (Float.pi / 180.0)
        let aspectRatio = Float(size.width) / Float(size.height)
        let imagePlaneHeight = tanf(fieldOfView / 2.0)
        let imagePlaneWidth = aspectRatio * imagePlaneHeight
        
        let camera = Camera(position: position,
                            right: right * imagePlaneWidth,
                            up: up * imagePlaneHeight,
                            forward: forward)
        return camera
    }
    
    static func setupLight() -> Light {
        return .areaLight(
            position: [0.0, 1.98, 0.0],
            forward: [0.0, -1.0, 0.0],
            right: [0.25, 0.0, 0.0],
            up: [0.0, 0.0, 0.25],
            color: [4.0, 4.0, 4.0]
        )
    }
}

extension Light {
    static func areaLight(position: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .areaLight
        light.position = position
        light.forward = forward
        light.right = right
        light.up = up
        light.color = color
        return light
    }
    
    static func sunLight(direction: SIMD3<Float>, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .sunlight
        light.direction = direction
        light.color = color
        return light
    }
    
    static func pointLight(position: SIMD3<Float>, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .pointlight
        light.position = position
        light.color = color
        return light
    }
    
    static func spotLight(position: SIMD3<Float>, direction: SIMD3<Float>, coneAngle: Float, color: SIMD3<Float>) -> Light {
        var light = Light()
        light.type = .spotlight
        light.position = position
        light.direction = direction
        light.coneAngle = coneAngle
        light.color = color
        return light
    }
}

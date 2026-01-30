//
//  GameViewController.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

#if os(macOS)
import Cocoa
import MetalKit

// Our macOS specific view controller
class GameViewController: NSViewController {

    var renderer: Renderer!
    var mtkView: MTKView!
    private var viewPresetPopup: NSPopUpButton?
    private var upscalerControl: NSSegmentedControl?
    private var scaleControl: NSPopUpButton?
    private var viewModeControl: NSSegmentedControl?
    var playerModelIndex: Int = 0

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }
        self.mtkView = mtkView

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        if !defaultDevice.supportsFamily(.metal4) {
            print("Metal 4 is not supported on this device")
            return
        }

        mtkView.device = defaultDevice

        guard let newRenderer = Renderer(metalView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)

        mtkView.delegate = renderer
        
        setupGestures()
        setupControls()
    }
    
    private func setupGestures() {
        guard let mtkView = mtkView else { return }
        
        let pan = NSPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        mtkView.addGestureRecognizer(pan)
        
        let magnify = NSMagnificationGestureRecognizer(target: self, action: #selector(handleMagnify(_:)))
        mtkView.addGestureRecognizer(magnify)
    }
    
    private func setupControls() {
        guard let mtkView = mtkView else { return }
        
        let container = NSVisualEffectView()
        container.material = .hudWindow
        container.blendingMode = .withinWindow
        container.state = .active
        container.translatesAutoresizingMaskIntoConstraints = false
        container.wantsLayer = true
        container.layer?.cornerRadius = 8
        
        let title = NSTextField(labelWithString: "View")
        title.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        
        let popup = NSPopUpButton()
        popup.addItems(withTitles: ["Free", "Front", "Back", "Left", "Right", "Top", "Bottom", "Isometric"])
        popup.selectItem(at: Renderer.ViewPreset.free.rawValue)
        popup.target = self
        popup.action = #selector(viewPresetChanged(_:))
        viewPresetPopup = popup
        
        // UpScaler control
        let upscalerLabel = NSTextField(labelWithString: "Upscaler:")
        upscalerLabel.font = NSFont.systemFont(ofSize: 12)
        
        let upscalerSegment = NSSegmentedControl(labels: ["OFF", "Spatial", "Temporal", "Denoise"], trackingMode: .selectOne, target: self, action: #selector(upscalerChanged(_:)))
        upscalerSegment.selectedSegment = 1 // Default Spatial
        upscalerControl = upscalerSegment
        
        // Samples per pixel control
        let samplesLabel = NSTextField(labelWithString: "Quality (spp):")
        samplesLabel.font = NSFont.systemFont(ofSize: 12)
        
        let samplesPopup = NSPopUpButton()
        samplesPopup.addItems(withTitles: ["1 - Fast", "2 - Balanced", "4 - Good", "8 - Best", "16 - High"])
        samplesPopup.selectItem(at: 1) // Default to 2
        samplesPopup.target = self
        samplesPopup.action = #selector(samplesChanged(_:))

        // Accumulation weight control
        let accumulationLabel = NSTextField(labelWithString: "Accumulation:")
        accumulationLabel.font = NSFont.systemFont(ofSize: 12)
        
        let accumulationSlider = NSSlider(value: 0.9, minValue: 0.0, maxValue: 0.95, target: self, action: #selector(accumulationChanged(_:)))
        accumulationSlider.isContinuous = true
        accumulationSlider.controlSize = .small
        accumulationSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true

        // Motion-adaptive accumulation controls
        let motionAccumCheckbox = NSButton(checkboxWithTitle: "Motion Accum", target: self, action: #selector(motionAccumulationToggled(_:)))
        motionAccumCheckbox.state = (renderer?.useMotionAdaptiveAccumulation ?? true) ? .on : .off
        
        let motionMinLabel = NSTextField(labelWithString: "Motion Min Weight:")
        motionMinLabel.font = NSFont.systemFont(ofSize: 12)
        let motionMinSlider = NSSlider(value: Double(renderer?.motionAccumulationMinWeight ?? 0.1), minValue: 0.0, maxValue: 0.95, target: self, action: #selector(motionAccumulationMinChanged(_:)))
        motionMinSlider.isContinuous = true
        motionMinSlider.controlSize = .small
        motionMinSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        
        let motionLowLabel = NSTextField(labelWithString: "Motion Low (px):")
        motionLowLabel.font = NSFont.systemFont(ofSize: 12)
        let motionLowSlider = NSSlider(value: Double(renderer?.motionAccumulationLowThresholdPixels ?? 0.5), minValue: 0.0, maxValue: 10.0, target: self, action: #selector(motionAccumulationLowChanged(_:)))
        motionLowSlider.isContinuous = true
        motionLowSlider.controlSize = .small
        motionLowSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        
        let motionHighLabel = NSTextField(labelWithString: "Motion High (px):")
        motionHighLabel.font = NSFont.systemFont(ofSize: 12)
        let motionHighSlider = NSSlider(value: Double(renderer?.motionAccumulationHighThresholdPixels ?? 4.0), minValue: 0.0, maxValue: 10.0, target: self, action: #selector(motionAccumulationHighChanged(_:)))
        motionHighSlider.isContinuous = true
        motionHighSlider.controlSize = .small
        motionHighSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        
        
        // Bounces control
        let bouncesLabel = NSTextField(labelWithString: "Ray Bounces:")
        bouncesLabel.font = NSFont.systemFont(ofSize: 12)
        
        let bouncesPopup = NSPopUpButton()
        bouncesPopup.addItems(withTitles: ["1", "2", "3", "4", "5"])
        bouncesPopup.selectItem(at: 1) // Default to 2
        bouncesPopup.target = self
        bouncesPopup.action = #selector(bouncesChanged(_:))
        
        // Render scale control
        let scaleLabel = NSTextField(labelWithString: "Render Scale:")
        scaleLabel.font = NSFont.systemFont(ofSize: 12)
        
        let scalePopup = NSPopUpButton()
        scalePopup.addItems(withTitles: ["34% (Performance)", "50%", "67% (Balanced)", "75%", "100% (Native)"])
        scalePopup.selectItem(at: 2) // Default to 67%
        scalePopup.target = self
        scalePopup.action = #selector(renderScaleChanged(_:))
        scaleControl = scalePopup
        
        // Camera Mode control
        let modeLabel = NSTextField(labelWithString: "Camera Mode:")
        modeLabel.font = NSFont.systemFont(ofSize: 12)
        
        let modeSegment = NSSegmentedControl(labels: ["World", "TPS"], trackingMode: .selectOne, target: self, action: #selector(viewModeChanged(_:)))
        modeSegment.selectedSegment = 0 // Default World
        viewModeControl = modeSegment
        
        // Debug Texture control
        let debugLabel = NSTextField(labelWithString: "Debug View:")
        debugLabel.font = NSFont.systemFont(ofSize: 12)
        
        let debugPopup = NSPopUpButton()
        debugPopup.addItems(withTitles: ["None", "Base Color", "Normal", "Roughness", "Metallic", "AO", "Emission", "Motion"])
        debugPopup.selectItem(at: 0)
        debugPopup.target = self
        debugPopup.action = #selector(debugTextureChanged(_:))

        // Shading Mode control
        let shadingLabel = NSTextField(labelWithString: "Shading:")
        shadingLabel.font = NSFont.systemFont(ofSize: 12)
        
        let shadingSegment = NSSegmentedControl(labels: ["PBR", "Legacy"], trackingMode: .selectOne, target: self, action: #selector(shadingChanged(_:)))
        shadingSegment.selectedSegment = 0
        
        // Light Intensity control
        let lightLabel = NSTextField(labelWithString: "Light Intensity:")
        lightLabel.font = NSFont.systemFont(ofSize: 12)
        
        let lightSlider = NSSlider(value: 4.0, minValue: 0.0, maxValue: 50.0, target: self, action: #selector(lightIntensityChanged(_:)))
        lightSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        
        let hint = NSTextField(labelWithString: "Drag: Orbit  Pinch: Zoom  M: Toggle MetalFX")
        hint.font = NSFont.systemFont(ofSize: 11)
        hint.textColor = NSColor.secondaryLabelColor
        
        let stack = NSStackView(views: [title, popup, modeLabel, modeSegment, debugLabel, debugPopup, shadingLabel, shadingSegment, samplesLabel, samplesPopup, accumulationLabel, accumulationSlider, motionAccumCheckbox, motionMinLabel, motionMinSlider, motionLowLabel, motionLowSlider, motionHighLabel, motionHighSlider, bouncesLabel, bouncesPopup, lightLabel, lightSlider, upscalerLabel, upscalerSegment, scaleLabel, scalePopup, hint])
        stack.orientation = .vertical
        stack.spacing = 6
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false
        
        container.addSubview(stack)
        mtkView.addSubview(container)
        
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 10),
            stack.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -10),
            stack.topAnchor.constraint(equalTo: container.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -8),
            
            container.leadingAnchor.constraint(equalTo: mtkView.leadingAnchor, constant: 12),
            container.topAnchor.constraint(equalTo: mtkView.topAnchor, constant: 12)
        ])
    }
    
    @objc private func viewPresetChanged(_ sender: NSPopUpButton) {
        guard let renderer = renderer else { return }
        let index = sender.indexOfSelectedItem
        if let preset = Renderer.ViewPreset(rawValue: index) {
            renderer.applyViewPreset(preset)
        }
    }
    
    @objc private func upscalerChanged(_ sender: NSSegmentedControl) {
        guard let renderer = renderer else { return }
        switch sender.selectedSegment {
        case 0:
            renderer.isMetalFXEnabled = false
            renderer.useTemporalScaler = false
            renderer.useTemporalDenoiser = false
        case 1:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = false
            renderer.useTemporalDenoiser = false
        case 2:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = true
            renderer.useTemporalDenoiser = false
        case 3:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = false
            renderer.useTemporalDenoiser = true
        default:
            break
        }
        scaleControl?.isEnabled = renderer.isMetalFXEnabled
    }
    
    @objc private func samplesChanged(_ sender: NSPopUpButton) {
        guard let renderer = renderer else { return }
        let samplesPerPixel: Int32
        switch sender.indexOfSelectedItem {
        case 0: samplesPerPixel = 1
        case 1: samplesPerPixel = 2
        case 2: samplesPerPixel = 4
        case 3: samplesPerPixel = 8
        case 4: samplesPerPixel = 16
        default: samplesPerPixel = 4
        }
        renderer.samplesPerPixel = samplesPerPixel
    }
    
    @objc private func bouncesChanged(_ sender: NSPopUpButton) {
        guard let renderer = renderer else { return }
        renderer.maxBounces = Int32(sender.indexOfSelectedItem + 1)
    }
    
    @objc private func renderScaleChanged(_ sender: NSPopUpButton) {
        guard let renderer = renderer else { return }
        let scale: Float
        switch sender.indexOfSelectedItem {
        case 0: scale = 0.34
        case 1: scale = 0.5
        case 2: scale = 0.67
        case 3: scale = 0.75
        case 4: scale = 1.0
        default: scale = 0.67
        }
        renderer.renderScale = scale
    }

    @objc private func viewModeChanged(_ sender: NSSegmentedControl) {
        guard let renderer = renderer else { return }
        if sender.selectedSegment == 0 {
            renderer.viewMode = .world
        } else {
            renderer.viewMode = .tps
            // Snap to back of character
            if renderer.scene.models.indices.contains(renderer.playerModelIndex) {
                 let player = renderer.scene.models[renderer.playerModelIndex]
                 renderer.cameraAzimuth = player.rotation.y
                 renderer.cameraElevation = 0.2
                 renderer.cameraDistance = 5.0
            }
        }
    }
    
    @objc private func debugTextureChanged(_ sender: NSPopUpButton) {
        guard let renderer = renderer else { return }
        renderer.debugTextureMode = Int32(sender.indexOfSelectedItem)
    }

    @objc private func shadingChanged(_ sender: NSSegmentedControl) {
        guard let renderer = renderer else { return }
        // Map selected segment to ShadingMode enum safely
        if let mode = Renderer.ShadingMode(rawValue: sender.selectedSegment) {
            renderer.shadingMode = mode
        }
    }
    
    @objc private func lightIntensityChanged(_ sender: NSSlider) {
        guard let renderer = renderer else { return }
        renderer.setLightIntensity(sender.floatValue)
    }

    @objc private func accumulationChanged(_ sender: NSSlider) {
        guard let renderer = renderer else { return }
        renderer.accumulationWeight = sender.floatValue
    }
    
    @objc private func motionAccumulationToggled(_ sender: NSButton) {
        renderer?.useMotionAdaptiveAccumulation = (sender.state == .on)
    }
    
    @objc private func motionAccumulationMinChanged(_ sender: NSSlider) {
        renderer?.motionAccumulationMinWeight = sender.floatValue
    }
    
    @objc private func motionAccumulationLowChanged(_ sender: NSSlider) {
        renderer?.motionAccumulationLowThresholdPixels = sender.floatValue
    }
    
    @objc private func motionAccumulationHighChanged(_ sender: NSSlider) {
        renderer?.motionAccumulationHighThresholdPixels = sender.floatValue
    }
    
    
    override var acceptsFirstResponder: Bool { return true }
    
    override func keyDown(with event: NSEvent) {
        guard let renderer = renderer else { return }
        
        // Movement speed
        let speed: Float = 0.5
        let rotationSpeed: Float = 0.1
        
        switch event.keyCode {
        case 13: // W
            renderer.scene.moveModel(index: playerModelIndex, forward: speed, right: 0)
        case 1: // S
            renderer.scene.moveModel(index: playerModelIndex, forward: -speed, right: 0)
        case 0: // A
            renderer.scene.moveModel(index: playerModelIndex, forward: 0, right: -speed)
        case 2: // D
            renderer.scene.moveModel(index: playerModelIndex, forward: 0, right: speed)
        case 123: // Left Arrow
            renderer.scene.rotateModel(index: playerModelIndex, angle: rotationSpeed)
        case 124: // Right Arrow
            renderer.scene.rotateModel(index: playerModelIndex, angle: -rotationSpeed)
        case 126: // Up Arrow
             renderer.scene.moveModel(index: playerModelIndex, forward: speed, right: 0)
        case 125: // Down Arrow
             renderer.scene.moveModel(index: playerModelIndex, forward: -speed, right: 0)
        default:
            super.keyDown(with: event)
        }
    }
    
    @objc private func handlePan(_ gesture: NSPanGestureRecognizer) {
        guard let renderer = renderer, let mtkView = mtkView else { return }
        let translation = gesture.translation(in: mtkView)
        renderer.orbit(deltaX: Float(translation.x), deltaY: Float(translation.y))
        gesture.setTranslation(.zero, in: mtkView)
        viewPresetPopup?.selectItem(at: Renderer.ViewPreset.free.rawValue)
    }
    
    @objc private func handleMagnify(_ gesture: NSMagnificationGestureRecognizer) {
        guard let renderer = renderer else { return }
        renderer.zoom(delta: Float(gesture.magnification))
        gesture.magnification = 0
        viewPresetPopup?.selectItem(at: Renderer.ViewPreset.free.rawValue)
    }
}
#elseif os(iOS)
import UIKit
import MetalKit

// Our iOS specific view controller (also used for Mac Catalyst)
class GameViewController: UIViewController {

    private let viewPresetTitles = ["Free", "Front", "Back", "Left", "Right", "Top", "Bottom", "Isometric"]
    private var selectedViewPresetIndex = Renderer.ViewPreset.free.rawValue

    var renderer: Renderer!
    var mtkView: MTKView!
    var playerModelIndex: Int = 0
    private var viewPresetButton: UIButton?
    private var upscalerControl: UISegmentedControl?
    private var scaleControl: UISegmentedControl?
    private var viewModeControl: UISegmentedControl?
    
    // UI Elements for visibility toggling
    private var controlsContainer: UIView?
    private var controlsToggleBtn: UIButton?
    
    // Motion Control Containers
    private var motionMinContainer: UIView?
    private var motionLowContainer: UIView?
    private var motionHighContainer: UIView?
    

    
    // Movement Control
    private var displayLink: CADisplayLink?
    private var joystickContainer: UIView?
    private var joystickStick: UIView?
    private var joystickCenter: CGPoint = .zero
    private var activeJoystickTouch: UITouch?
    
    private var currentInput: SIMD2<Float> = .zero // x: rotation, y: speed

    override func loadView() {
        mtkView = MTKView(frame: .zero)
        mtkView.isMultipleTouchEnabled = true
        view = mtkView
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }
        self.mtkView = mtkView

        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        if !defaultDevice.supportsFamily(.metal4) {
            print("Metal 4 is not supported on this device")
            return
        }

        mtkView.device = defaultDevice

        guard let newRenderer = Renderer(metalView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
//        if let exportURL = renderer.runSkinningDebugTestsExportingToTmp() {
//            print("CPU skinning export dir:", exportURL.path)
//        }
        mtkView.delegate = renderer
        
        setupGestures()
        setupControls()
        setupJoystick()
    }

    private func setupGestures() {
        guard let mtkView = mtkView else { return }

        let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        pan.maximumNumberOfTouches = 1
        mtkView.addGestureRecognizer(pan)

        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        mtkView.addGestureRecognizer(pinch)
    }

    private func setupControls() {
        guard let mtkView = mtkView else { return }

        // Create the toggle button
        let toggleBtn = UIButton(type: .system)
        toggleBtn.setImage(UIImage(systemName: "line.3.horizontal"), for: .normal)
        toggleBtn.tintColor = .label
        toggleBtn.backgroundColor = .systemBackground.withAlphaComponent(0.7)
        toggleBtn.layer.cornerRadius = 8
        toggleBtn.translatesAutoresizingMaskIntoConstraints = false
        toggleBtn.addTarget(self, action: #selector(toggleControls(_:)), for: .touchUpInside)
        mtkView.addSubview(toggleBtn)
        self.controlsToggleBtn = toggleBtn

        // Container for all other controls
        let container = UIVisualEffectView(effect: UIBlurEffect(style: .systemMaterial))
        container.translatesAutoresizingMaskIntoConstraints = false
        container.layer.cornerRadius = 10
        container.clipsToBounds = true
        self.controlsContainer = container

        let title = UILabel()
        title.text = "View"
        title.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
        title.textColor = .label

        let button = UIButton(type: .system)
        button.setTitle(viewPresetTitles[selectedViewPresetIndex], for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 13, weight: .semibold)
        button.contentEdgeInsets = UIEdgeInsets(top: 4, left: 8, bottom: 4, right: 8)
        button.showsMenuAsPrimaryAction = true
        button.menu = buildViewPresetMenu(selectedIndex: selectedViewPresetIndex)
        viewPresetButton = button

        // UpScaler control
        let upscalerContainer = UIStackView()
        upscalerContainer.axis = .horizontal
        upscalerContainer.spacing = 8
        upscalerContainer.alignment = .center
        
        let upscalerLabel = UILabel()
        upscalerLabel.text = "Upscaler"
        upscalerLabel.font = UIFont.systemFont(ofSize: 13)
        upscalerLabel.textColor = .label
        
        let upscalerSegment = UISegmentedControl(items: ["OFF", "Spatial", "Temporal", "Denoise"])
        upscalerSegment.selectedSegmentIndex = 1 // Default Spatial
        upscalerSegment.addTarget(self, action: #selector(upscalerChanged(_:)), for: .valueChanged)
        self.upscalerControl = upscalerSegment
        
        upscalerContainer.addArrangedSubview(upscalerLabel)
        upscalerContainer.addArrangedSubview(upscalerSegment)
        
        // Samples per pixel control
        let samplesContainer = UIStackView()
        samplesContainer.axis = .horizontal
        samplesContainer.spacing = 8
        samplesContainer.alignment = .center
        
        let samplesLabel = UILabel()
        samplesLabel.text = "Quality: 2 spp"
        samplesLabel.font = UIFont.systemFont(ofSize: 13)
        samplesLabel.textColor = .label
        samplesLabel.tag = 999 // To find and update later
        
        let samplesSegmented = UISegmentedControl(items: ["1", "2", "4", "8", "16"])
        samplesSegmented.selectedSegmentIndex = 1 // Default to 2
        samplesSegmented.addTarget(self, action: #selector(samplesChanged(_:)), for: .valueChanged)
        
        samplesContainer.addArrangedSubview(samplesLabel)
        samplesContainer.addArrangedSubview(samplesSegmented)

        // Accumulation control
        let accumulationContainer = UIStackView()
        accumulationContainer.axis = .horizontal
        accumulationContainer.spacing = 8
        accumulationContainer.alignment = .center
        
        let accumulationLabel = UILabel()
        accumulationLabel.text = "Accum: 0.90"
        accumulationLabel.font = UIFont.systemFont(ofSize: 13)
        accumulationLabel.textColor = .label
        accumulationLabel.tag = 995 // For update
        
        let accumulationSlider = UISlider()
        accumulationSlider.minimumValue = 0.0
        accumulationSlider.maximumValue = 0.95
        accumulationSlider.value = 0.9
        accumulationSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        accumulationSlider.addTarget(self, action: #selector(accumulationChanged(_:)), for: .valueChanged)
        if let renderer = renderer {
            renderer.accumulationWeight = accumulationSlider.value
        }
        
        accumulationContainer.addArrangedSubview(accumulationLabel)
        accumulationContainer.addArrangedSubview(accumulationSlider)

        // Motion-adaptive accumulation (iOS)
        let motionAccumContainer = UIStackView()
        motionAccumContainer.axis = .vertical
        motionAccumContainer.spacing = 2
        motionAccumContainer.alignment = .leading
        
        let motionAccumLabel = UILabel()
        motionAccumLabel.text = "Motion Accum"
        motionAccumLabel.font = UIFont.systemFont(ofSize: 13)
        motionAccumLabel.textColor = .label
        
        let motionAccumSwitch = UISwitch()
        motionAccumSwitch.isOn = renderer?.useMotionAdaptiveAccumulation ?? true
        motionAccumSwitch.addTarget(self, action: #selector(motionAccumulationToggled(_:)), for: .valueChanged)
        
        motionAccumContainer.addArrangedSubview(motionAccumLabel)
        motionAccumContainer.addArrangedSubview(motionAccumSwitch)
        
        let motionMinContainer = UIStackView()
        motionMinContainer.axis = .vertical
        motionMinContainer.spacing = 2
        motionMinContainer.alignment = .leading
        
        let motionMinLabel = UILabel()
        let motionMinValue = renderer?.motionAccumulationMinWeight ?? 0.1
        motionMinLabel.text = String(format: "Motion Min: %.2f", motionMinValue)
        motionMinLabel.font = UIFont.systemFont(ofSize: 13)
        motionMinLabel.textColor = .label
        motionMinLabel.tag = 994
        
        let motionMinSlider = UISlider()
        motionMinSlider.minimumValue = 0.0
        motionMinSlider.maximumValue = 0.95
        motionMinSlider.value = motionMinValue
        motionMinSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        motionMinSlider.addTarget(self, action: #selector(motionAccumulationMinChanged(_:)), for: .valueChanged)
        
        motionMinContainer.addArrangedSubview(motionMinLabel)
        motionMinContainer.addArrangedSubview(motionMinSlider)
        
        self.motionMinContainer = motionMinContainer
        motionMinContainer.isHidden = !(renderer?.useMotionAdaptiveAccumulation ?? true)
        
        let motionLowContainer = UIStackView()
        motionLowContainer.axis = .vertical
        motionLowContainer.spacing = 2
        motionLowContainer.alignment = .leading
        
        let motionLowLabel = UILabel()
        let motionLowValue = renderer?.motionAccumulationLowThresholdPixels ?? 0.5
        motionLowLabel.text = String(format: "Motion Low: %.2fpx", motionLowValue)
        motionLowLabel.font = UIFont.systemFont(ofSize: 13)
        motionLowLabel.textColor = .label
        motionLowLabel.tag = 993
        
        let motionLowSlider = UISlider()
        motionLowSlider.minimumValue = 0.0
        motionLowSlider.maximumValue = 10.0
        motionLowSlider.value = motionLowValue
        motionLowSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        motionLowSlider.addTarget(self, action: #selector(motionAccumulationLowChanged(_:)), for: .valueChanged)
        
        motionLowContainer.addArrangedSubview(motionLowLabel)
        motionLowContainer.addArrangedSubview(motionLowSlider)
        
        self.motionLowContainer = motionLowContainer
        motionLowContainer.isHidden = !(renderer?.useMotionAdaptiveAccumulation ?? true)
        
        let motionHighContainer = UIStackView()
        motionHighContainer.axis = .vertical
        motionHighContainer.spacing = 2
        motionHighContainer.alignment = .leading
        
        let motionHighLabel = UILabel()
        let motionHighValue = renderer?.motionAccumulationHighThresholdPixels ?? 4.0
        motionHighLabel.text = String(format: "Motion High: %.2fpx", motionHighValue)
        motionHighLabel.font = UIFont.systemFont(ofSize: 13)
        motionHighLabel.textColor = .label
        motionHighLabel.tag = 992
        
        let motionHighSlider = UISlider()
        motionHighSlider.minimumValue = 0.0
        motionHighSlider.maximumValue = 10.0
        motionHighSlider.value = motionHighValue
        motionHighSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        motionHighSlider.addTarget(self, action: #selector(motionAccumulationHighChanged(_:)), for: .valueChanged)
        
        motionHighContainer.addArrangedSubview(motionHighLabel)
        motionHighContainer.addArrangedSubview(motionHighSlider)
        
        self.motionHighContainer = motionHighContainer
        motionHighContainer.isHidden = !(renderer?.useMotionAdaptiveAccumulation ?? true)
        
        
        // Bounces control
        let bouncesContainer = UIStackView()
        bouncesContainer.axis = .horizontal
        bouncesContainer.spacing = 8
        bouncesContainer.alignment = .center
        
        let bouncesLabel = UILabel()
        bouncesLabel.text = "Bounces: 2"
        bouncesLabel.font = UIFont.systemFont(ofSize: 13)
        bouncesLabel.textColor = .label
        bouncesLabel.tag = 998 // To find and update later
        
        let bouncesSegmented = UISegmentedControl(items: ["1", "2", "3", "4"])
        bouncesSegmented.selectedSegmentIndex = 1 // Default to 2
        bouncesSegmented.addTarget(self, action: #selector(bouncesChanged(_:)), for: .valueChanged)
        
        bouncesContainer.addArrangedSubview(bouncesLabel)
        bouncesContainer.addArrangedSubview(bouncesSegmented)
        
        // Light Intensity control
        let lightContainer = UIStackView()
        lightContainer.axis = .horizontal
        lightContainer.spacing = 8
        lightContainer.alignment = .center
        
        let initialIntensity: Float = 15.0
        let lightLabel = UILabel()
        lightLabel.text = "Light: \(initialIntensity)"
        lightLabel.font = UIFont.systemFont(ofSize: 13)
        lightLabel.textColor = .label
        lightLabel.tag = 996 // For update
        
        let lightSlider = UISlider()
        lightSlider.minimumValue = 0.0
        lightSlider.maximumValue = 50.0
        lightSlider.value = initialIntensity
        lightSlider.widthAnchor.constraint(equalToConstant: 120).isActive = true
        lightSlider.addTarget(self, action: #selector(lightIntensityChanged(_:)), for: .valueChanged)
        if let renderer = renderer {
            renderer.setLightIntensity(initialIntensity)
        }
        lightContainer.addArrangedSubview(lightLabel)
        lightContainer.addArrangedSubview(lightSlider)
        
        // Render scale control (for MetalFX)
        let scaleContainer = UIStackView()
        scaleContainer.axis = .horizontal
        scaleContainer.spacing = 8
        scaleContainer.alignment = .center
        
        let scaleLabel = UILabel()
        scaleLabel.text = "Render Scale: 67%"
        scaleLabel.font = UIFont.systemFont(ofSize: 13)
        scaleLabel.textColor = .label
        scaleLabel.tag = 997 // To find and update later
        
        let scaleSegmented = UISegmentedControl(items: ["34%", "50%", "67%", "75%", "100%"])
        scaleSegmented.selectedSegmentIndex = 2 // Default to 67%
        scaleSegmented.addTarget(self, action: #selector(renderScaleChanged(_:)), for: .valueChanged)
        self.scaleControl = scaleSegmented
        
        scaleContainer.addArrangedSubview(scaleLabel)
        scaleContainer.addArrangedSubview(scaleSegmented)
        
        let hint = UILabel()
        hint.text = "Drag: Orbit  Pinch: Zoom"
        hint.font = UIFont.systemFont(ofSize: 11)
        hint.textColor = .secondaryLabel

        // Camera Mode control
        let modeContainer = UIStackView()
        modeContainer.axis = .horizontal
        modeContainer.spacing = 8
        modeContainer.alignment = .center
        
        let modeLabel = UILabel()
        modeLabel.text = "Mode"
        modeLabel.font = UIFont.systemFont(ofSize: 13)
        modeLabel.textColor = .label
        
        let modeSegment = UISegmentedControl(items: ["World", "TPS"])
        modeSegment.selectedSegmentIndex = 0 // Default World
        modeSegment.addTarget(self, action: #selector(viewModeChanged(_:)), for: .valueChanged)
        self.viewModeControl = modeSegment
        
        modeContainer.addArrangedSubview(modeLabel)
        modeContainer.addArrangedSubview(modeSegment)

        // Debug Texture Control (iOS)
        let debugContainer = UIStackView()
        debugContainer.axis = .horizontal
        debugContainer.spacing = 8
        debugContainer.alignment = .center
        
        let debugLabel = UILabel()
        debugLabel.text = "Debug"
        debugLabel.font = UIFont.systemFont(ofSize: 13)
        debugLabel.textColor = .label
        
        let debugButton = UIButton(type: .system)
        debugButton.setTitle("None", for: .normal)
        debugButton.titleLabel?.font = UIFont.systemFont(ofSize: 13, weight: .semibold)
        debugButton.contentEdgeInsets = UIEdgeInsets(top: 4, left: 8, bottom: 4, right: 8)
        debugButton.showsMenuAsPrimaryAction = true
        
        let debugTitles = ["None", "Base Color", "Normal", "Roughness", "Metallic", "AO", "Emission", "Motion"]
        let debugActions = debugTitles.enumerated().map { index, title in
            UIAction(title: title, state: index == 0 ? .on : .off) { [weak self] _ in
                guard let self = self else { return }
                self.renderer?.debugTextureMode = Int32(index)
                debugButton.setTitle(title, for: .normal)
            }
        }
        debugButton.menu = UIMenu(children: debugActions)
        
        debugContainer.addArrangedSubview(debugLabel)
        debugContainer.addArrangedSubview(debugButton)

        // Shading Mode control (iOS)
        let shadingContainer = UIStackView()
        shadingContainer.axis = .horizontal
        shadingContainer.spacing = 8
        shadingContainer.alignment = .center
        
        let shadingLabel = UILabel()
        shadingLabel.text = "Shading"
        shadingLabel.font = UIFont.systemFont(ofSize: 13)
        shadingLabel.textColor = .label
        
        let shadingSegment = UISegmentedControl(items: ["PBR", "Legacy"])
        shadingSegment.selectedSegmentIndex = 0
        shadingSegment.addTarget(self, action: #selector(shadingChanged(_:)), for: .valueChanged)
        
        shadingContainer.addArrangedSubview(shadingLabel)
        shadingContainer.addArrangedSubview(shadingSegment)

        let stack = UIStackView(arrangedSubviews: [title, button, modeContainer, debugContainer, shadingContainer, samplesContainer, accumulationContainer, bouncesContainer, lightContainer, upscalerContainer, scaleContainer, motionAccumContainer, motionMinContainer, motionLowContainer, motionHighContainer, hint])
        stack.axis = .vertical
        stack.spacing = 6
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false

        let scrollView = UIScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.alwaysBounceVertical = true
        scrollView.showsVerticalScrollIndicator = true

        container.contentView.addSubview(scrollView)
        scrollView.addSubview(stack)
        mtkView.addSubview(container)

        let fitHeight = container.heightAnchor.constraint(equalTo: stack.heightAnchor, constant: 16)
        fitHeight.priority = .defaultHigh
        let maxHeight = container.heightAnchor.constraint(lessThanOrEqualTo: mtkView.safeAreaLayoutGuide.heightAnchor, constant: -24)
        maxHeight.priority = .required

        NSLayoutConstraint.activate([
            // Button constraints
            toggleBtn.topAnchor.constraint(equalTo: mtkView.safeAreaLayoutGuide.topAnchor, constant: 12),
            toggleBtn.leadingAnchor.constraint(equalTo: mtkView.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            toggleBtn.widthAnchor.constraint(equalToConstant: 44),
            toggleBtn.heightAnchor.constraint(equalToConstant: 44),

            // Scroll view constraints within container
            scrollView.leadingAnchor.constraint(equalTo: container.contentView.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: container.contentView.trailingAnchor),
            scrollView.topAnchor.constraint(equalTo: container.contentView.topAnchor),
            scrollView.bottomAnchor.constraint(equalTo: container.contentView.bottomAnchor),

            // Stack constraints within scroll view
            stack.leadingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.leadingAnchor, constant: 10),
            stack.trailingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.trailingAnchor, constant: -10),
            stack.topAnchor.constraint(equalTo: scrollView.contentLayoutGuide.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: scrollView.contentLayoutGuide.bottomAnchor, constant: -8),
            stack.widthAnchor.constraint(equalTo: scrollView.frameLayoutGuide.widthAnchor, constant: -20),

            // Container constraints - attached to toggle button
            container.leadingAnchor.constraint(equalTo: toggleBtn.leadingAnchor),
            container.topAnchor.constraint(equalTo: toggleBtn.bottomAnchor, constant: 8),
            container.trailingAnchor.constraint(lessThanOrEqualTo: mtkView.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            container.bottomAnchor.constraint(lessThanOrEqualTo: mtkView.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            fitHeight,
            maxHeight
        ])
    }

    @objc private func toggleControls(_ sender: UIButton) {
        guard let container = controlsContainer else { return }
        
        UIView.animate(withDuration: 0.3) {
            container.isHidden.toggle()
            container.alpha = container.isHidden ? 0 : 1
        }

        joystickContainer?.isHidden = !container.isHidden
        
        let iconName = container.isHidden ? "eye" : "line.3.horizontal" // Eye to show, Menu to hide? Or just toggle.
        // Let's stick to the menu icon or maybe switch to 'xmark' when open.
        // Actually, just keeping the menu icon is often fine, or toggling opacity.
        // Let's maybe switch to X when open.
        let newIconName = container.isHidden ? "slider.horizontal.3" : "xmark"
        sender.setImage(UIImage(systemName: newIconName), for: .normal)
    }

    private func buildViewPresetMenu(selectedIndex: Int) -> UIMenu {
        let actions = viewPresetTitles.enumerated().compactMap { index, title -> UIAction? in
            guard let preset = Renderer.ViewPreset(rawValue: index) else { return nil }
            return UIAction(title: title, state: index == selectedIndex ? .on : .off) { [weak self] _ in
                guard let self = self else { return }
                self.renderer?.applyViewPreset(preset)
                self.setViewPresetSelection(index: index)
            }
        }
        return UIMenu(children: actions)
    }

    private func setViewPresetSelection(index: Int) {
        selectedViewPresetIndex = index
        viewPresetButton?.setTitle(viewPresetTitles[index], for: .normal)
        viewPresetButton?.menu = buildViewPresetMenu(selectedIndex: index)
    }
    
    @objc private func upscalerChanged(_ sender: UISegmentedControl) {
        guard let renderer = renderer else { return }
        switch sender.selectedSegmentIndex {
        case 0:
            renderer.isMetalFXEnabled = false
            renderer.useTemporalScaler = false
            renderer.useTemporalDenoiser = false
        case 1:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = false
            renderer.useTemporalDenoiser = false
        case 2:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = true
            renderer.useTemporalDenoiser = false
        case 3:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = false
            renderer.useTemporalDenoiser = true
        default:
            break
        }
        scaleControl?.isEnabled = renderer.isMetalFXEnabled
    }
    
    @objc private func samplesChanged(_ sender: UISegmentedControl) {
        let samplesPerPixel: Int32
        switch sender.selectedSegmentIndex {
        case 0: samplesPerPixel = 1
        case 1: samplesPerPixel = 2
        case 2: samplesPerPixel = 4
        case 3: samplesPerPixel = 8
        case 4: samplesPerPixel = 16
        default: samplesPerPixel = 4
        }
        renderer?.samplesPerPixel = samplesPerPixel
        
        // Update label
        if let container = sender.superview,
           let label = container.viewWithTag(999) as? UILabel {
            label.text = "Quality: \(samplesPerPixel) spp"
        }
    }
    
    @objc private func bouncesChanged(_ sender: UISegmentedControl) {
        let bounces: Int32 = Int32(sender.selectedSegmentIndex + 1)
        renderer?.maxBounces = bounces
        
        // Update label
        if let container = sender.superview,
           let label = container.viewWithTag(998) as? UILabel {
            label.text = "Bounces: \(bounces)"
        }
    }
    
    @objc private func lightIntensityChanged(_ sender: UISlider) {
        guard let renderer = renderer else { return }
        renderer.setLightIntensity(sender.value)
        
        // Update label
        if let container = sender.superview,
           let label = container.viewWithTag(996) as? UILabel {
            label.text = String(format: "Light: %.1f", sender.value)
        }
    }

    @objc private func accumulationChanged(_ sender: UISlider) {
        renderer?.accumulationWeight = sender.value
        
        // Update label
        if let container = sender.superview,
           let label = container.viewWithTag(995) as? UILabel {
            label.text = String(format: "Accum: %.2f", sender.value)
        }
    }
    
    @objc private func motionAccumulationToggled(_ sender: UISwitch) {
        renderer?.useMotionAdaptiveAccumulation = sender.isOn
        
        UIView.animate(withDuration: 0.3) {
            self.motionMinContainer?.isHidden = !sender.isOn
            self.motionLowContainer?.isHidden = !sender.isOn
            self.motionHighContainer?.isHidden = !sender.isOn
            self.motionMinContainer?.alpha = sender.isOn ? 1 : 0
            self.motionLowContainer?.alpha = sender.isOn ? 1 : 0
            self.motionHighContainer?.alpha = sender.isOn ? 1 : 0
            self.view.layoutIfNeeded()
        }
    }
    
    @objc private func motionAccumulationMinChanged(_ sender: UISlider) {
        renderer?.motionAccumulationMinWeight = sender.value
        if let container = sender.superview,
           let label = container.viewWithTag(994) as? UILabel {
            label.text = String(format: "Motion Min: %.2f", sender.value)
        }
    }
    
    @objc private func motionAccumulationLowChanged(_ sender: UISlider) {
        renderer?.motionAccumulationLowThresholdPixels = sender.value
        if let container = sender.superview,
           let label = container.viewWithTag(993) as? UILabel {
            label.text = String(format: "Motion Low: %.2fpx", sender.value)
        }
    }
    
    @objc private func motionAccumulationHighChanged(_ sender: UISlider) {
        renderer?.motionAccumulationHighThresholdPixels = sender.value
        if let container = sender.superview,
           let label = container.viewWithTag(992) as? UILabel {
            label.text = String(format: "Motion High: %.2fpx", sender.value)
        }
    }
    
    
    @objc private func renderScaleChanged(_ sender: UISegmentedControl) {
        let scale: Float
        let labelText: String
        switch sender.selectedSegmentIndex {
        case 0:
            scale = 0.34
            labelText = "34%"
        case 1:
            scale = 0.5
            labelText = "50%"
        case 2:
            scale = 0.67
            labelText = "67%"
        case 3:
            scale = 0.75
            labelText = "75%"
        case 4:
            scale = 1.0
            labelText = "100%"
        default: 
            scale = 0.67
            labelText = "67%"
        }
        renderer?.renderScale = scale
        
        // Update label
        if let container = sender.superview,
           let label = container.viewWithTag(997) as? UILabel {
            label.text = "Render Scale: \(labelText)"
        }
    }

    @objc private func shadingChanged(_ sender: UISegmentedControl) {
        guard let renderer = renderer else { return }
        if let mode = ShadingMode(rawValue: sender.selectedSegmentIndex) {
            renderer.shadingMode = mode
        }
    }
    
    @objc private func viewModeChanged(_ sender: UISegmentedControl) {
        guard let renderer = renderer else { return }
        if sender.selectedSegmentIndex == 0 {
            renderer.viewMode = .world
        } else {
            renderer.viewMode = .tps
            // Snap to back of character
            if renderer.scene.models.indices.contains(renderer.playerModelIndex) {
                 let player = renderer.scene.models[renderer.playerModelIndex]
                 renderer.cameraAzimuth = player.rotation.y
                 renderer.cameraElevation = 0.2
                 renderer.cameraDistance = 3.0
            }
        }
    }

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let renderer = renderer, let mtkView = mtkView else { return }
        let translation = gesture.translation(in: mtkView)
        renderer.orbit(deltaX: Float(translation.x), deltaY: Float(translation.y))
        gesture.setTranslation(.zero, in: mtkView)
        setViewPresetSelection(index: Renderer.ViewPreset.free.rawValue)
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let renderer = renderer else { return }
        let delta = Float(gesture.scale - 1.0)
        renderer.zoom(delta: delta)
        gesture.scale = 1.0
        setViewPresetSelection(index: Renderer.ViewPreset.free.rawValue)
    }
    
    // MARK: - Virtual Joystick Implementation
    
    private func setupJoystick() {
        guard let mtkView = mtkView else { return }
        
        let containerSize: CGFloat = 120
        let stickSize: CGFloat = 50
        
        let container = UIView()
        container.translatesAutoresizingMaskIntoConstraints = false
        container.backgroundColor = UIColor.black.withAlphaComponent(0.3)
        container.layer.cornerRadius = containerSize / 2
        container.layer.borderWidth = 2
        container.layer.borderColor = UIColor.white.withAlphaComponent(0.3).cgColor
        
        mtkView.addSubview(container)
        self.joystickContainer = container
        
        let stick = UIView()
        stick.translatesAutoresizingMaskIntoConstraints = false
        stick.backgroundColor = UIColor.white.withAlphaComponent(0.8)
        stick.layer.cornerRadius = stickSize / 2
        stick.isUserInteractionEnabled = false // Container handles touches
        
        container.addSubview(stick)
        self.joystickStick = stick
        
        NSLayoutConstraint.activate([
            container.widthAnchor.constraint(equalToConstant: containerSize),
            container.heightAnchor.constraint(equalToConstant: containerSize),
            container.leadingAnchor.constraint(equalTo: mtkView.safeAreaLayoutGuide.leadingAnchor, constant: 40),
            container.bottomAnchor.constraint(equalTo: mtkView.safeAreaLayoutGuide.bottomAnchor, constant: -40),
            
            stick.widthAnchor.constraint(equalToConstant: stickSize),
            stick.heightAnchor.constraint(equalToConstant: stickSize),
            stick.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            stick.centerYAnchor.constraint(equalTo: container.centerYAnchor)
        ])
        
        // Touch handling
        // Since UIView touches are tricky in a controller, we can use a custom view or gesture recognizers.
        // A PanGesture is easiest for a joystick.
        
        let pan = UIPanGestureRecognizer(target: self, action: #selector(handleJoystickPan(_:)))
        container.addGestureRecognizer(pan)
        container.isHidden = !(controlsContainer?.isHidden ?? true)
    }
    
    @objc private func handleJoystickPan(_ gesture: UIPanGestureRecognizer) {
        guard let container = joystickContainer, let stick = joystickStick else { return }
        
        let maxRadius = (container.bounds.width - stick.bounds.width) / 2
        
        switch gesture.state {
        case .began, .changed:
            let translation = gesture.translation(in: container)
            let center = CGPoint(x: container.bounds.width / 2, y: container.bounds.height / 2)
            
            // Calculate distance from center
            let vector = SIMD2<Float>(Float(translation.x), Float(translation.y))
            let distance = simd_length(vector)
            
            var clampedVector = vector
            if distance > Float(maxRadius) {
                clampedVector = simd_normalize(vector) * Float(maxRadius)
            }
            
            // Update stick position
            stick.transform = CGAffineTransform(translationX: CGFloat(clampedVector.x), y: CGFloat(clampedVector.y))
            
            // Normalize input (-1.0 to 1.0)
            // Invert Y because screen Y is down, but we want Up to be positive forward
            let normX = clampedVector.x / Float(maxRadius)
            let normY = -(clampedVector.y / Float(maxRadius))
            currentInput = SIMD2<Float>(normX, normY)
            
            if displayLink == nil {
                startMovementLoop()
            }
            
        case .ended, .cancelled:
            UIView.animate(withDuration: 0.2) {
                stick.transform = .identity
            }
            currentInput = .zero
            stopMovementLoop()
            
        default:
            break
        }
    }
    
    private func startMovementLoop() {
        displayLink = CADisplayLink(target: self, selector: #selector(updateMovement))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    private func stopMovementLoop() {
        displayLink?.invalidate()
        displayLink = nil
    }
    

        
    @objc private func updateMovement() {
        guard currentInput != .zero else { return }
        guard let renderer = renderer else { return }
        
        // Directional Movement Logic
        // 1. Calculate Joystick Angle
        // atan2(x, y) results in 0 at Y+, PI/2 at X+, etc. which matches our need if we map to World space relative to Camera.
        // Joystick Up (Y+) -> 0 radians.
        // 3. Target Rotation = CameraAngle + JoystickAngle
        // Fix: Invert X for correct Left/Right mapping (atan2 takes y, x usually or we just flip sign)
        // Fix: Add PI to fix 180 degree offset (Model facing backwards)
        let joystickAngle = atan2(-currentInput.x, currentInput.y)
        
        let cameraAzimuth = renderer.scene.cameraAzimuth
        let targetRotation = cameraAzimuth + joystickAngle + Float.pi
        
        // 4. Update Model
        // Set rotation to face the direction
        renderer.scene.setModelRotation(index: playerModelIndex, angle: targetRotation)
        
        // Move forward
        // Since we rotated 180 degrees (added PI), our Local Forward (-Z) is now World Backwards relative to visual Front (+Z).
        // So we need to invert the speed to move in the visual "Forward" direction.
        let inputMagnitude = simd_length(currentInput)
        let moveSpeed: Float = 0.03 * inputMagnitude
        
        renderer.scene.moveModel(index: playerModelIndex, forward: -moveSpeed, right: 0)
    }
}
#endif

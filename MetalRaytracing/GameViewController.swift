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
        
        let upscalerSegment = NSSegmentedControl(labels: ["OFF", "Spatial", "Temporal"], trackingMode: .selectOne, target: self, action: #selector(upscalerChanged(_:)))
        upscalerSegment.selectedSegment = 1 // Default Spatial
        upscalerControl = upscalerSegment
        
        // Samples per pixel control
        let samplesLabel = NSTextField(labelWithString: "Quality (spp):")
        samplesLabel.font = NSFont.systemFont(ofSize: 12)
        
        let samplesPopup = NSPopUpButton()
        samplesPopup.addItems(withTitles: ["1 - Fast", "2 - Balanced", "4 - Good", "8 - Best"])
        samplesPopup.selectItem(at: 1) // Default to 2
        samplesPopup.target = self
        samplesPopup.action = #selector(samplesChanged(_:))
        
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
        scalePopup.addItems(withTitles: ["50%", "67% (Balanced)", "75%", "100% (Native)"])
        scalePopup.selectItem(at: 1) // Default to 67%
        scalePopup.target = self
        scalePopup.action = #selector(renderScaleChanged(_:))
        scaleControl = scalePopup
        
        let hint = NSTextField(labelWithString: "Drag: Orbit  Pinch: Zoom  M: Toggle MetalFX")
        hint.font = NSFont.systemFont(ofSize: 11)
        hint.textColor = NSColor.secondaryLabelColor
        
        let stack = NSStackView(views: [title, popup, samplesLabel, samplesPopup, bouncesLabel, bouncesPopup, upscalerLabel, upscalerSegment, scaleLabel, scalePopup, hint])
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
        case 1:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = false
        case 2:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = true
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
        case 0: scale = 0.5
        case 1: scale = 0.67
        case 2: scale = 0.75
        case 3: scale = 1.0
        default: scale = 0.67
        }
        renderer.renderScale = scale
    }
    
    override func keyDown(with event: NSEvent) {
        // "m" key handling removed as it doesn't map cleanly to 3-state logic anymore, or could cycle.
        // For now, removing to be safe and consistent with UI.
        super.keyDown(with: event)
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
    private var viewPresetButton: UIButton?
    private var upscalerControl: UISegmentedControl?
    private var scaleControl: UISegmentedControl?
    
    // UI Elements for visibility toggling
    private var controlsContainer: UIView?
    private var controlsToggleBtn: UIButton?

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

        mtkView.delegate = renderer

        setupGestures()
        setupControls()
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
        
        let upscalerSegment = UISegmentedControl(items: ["OFF", "Spatial", "Temporal"])
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
        
        let samplesSegmented = UISegmentedControl(items: ["1", "2", "4", "8"])
        samplesSegmented.selectedSegmentIndex = 1 // Default to 2
        samplesSegmented.addTarget(self, action: #selector(samplesChanged(_:)), for: .valueChanged)
        
        samplesContainer.addArrangedSubview(samplesLabel)
        samplesContainer.addArrangedSubview(samplesSegmented)
        
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
        
        // Render scale control (for MetalFX)
        let scaleContainer = UIStackView()
        scaleContainer.axis = .horizontal
        scaleContainer.spacing = 8
        scaleContainer.alignment = .center
        
        let scaleLabel = UILabel()
        scaleLabel.text = "UpScale Ratio: 67%"
        scaleLabel.font = UIFont.systemFont(ofSize: 13)
        scaleLabel.textColor = .label
        scaleLabel.tag = 997 // To find and update later
        
        let scaleSegmented = UISegmentedControl(items: ["50%", "67%", "75%", "100%"])
        scaleSegmented.selectedSegmentIndex = 1 // Default to 67%
        scaleSegmented.addTarget(self, action: #selector(renderScaleChanged(_:)), for: .valueChanged)
        self.scaleControl = scaleSegmented
        
        scaleContainer.addArrangedSubview(scaleLabel)
        scaleContainer.addArrangedSubview(scaleSegmented)
        
        let hint = UILabel()
        hint.text = "Drag: Orbit  Pinch: Zoom"
        hint.font = UIFont.systemFont(ofSize: 11)
        hint.textColor = .secondaryLabel

        let stack = UIStackView(arrangedSubviews: [title, button, samplesContainer, bouncesContainer, upscalerContainer, scaleContainer, hint])
        stack.axis = .vertical
        stack.spacing = 6
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false

        container.contentView.addSubview(stack)
        mtkView.addSubview(container)

        NSLayoutConstraint.activate([
            // Button constraints
            toggleBtn.topAnchor.constraint(equalTo: mtkView.safeAreaLayoutGuide.topAnchor, constant: 12),
            toggleBtn.leadingAnchor.constraint(equalTo: mtkView.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            toggleBtn.widthAnchor.constraint(equalToConstant: 44),
            toggleBtn.heightAnchor.constraint(equalToConstant: 44),

            // Stack constraints within container
            stack.leadingAnchor.constraint(equalTo: container.contentView.leadingAnchor, constant: 10),
            stack.trailingAnchor.constraint(equalTo: container.contentView.trailingAnchor, constant: -10),
            stack.topAnchor.constraint(equalTo: container.contentView.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: container.contentView.bottomAnchor, constant: -8),

            // Container constraints - attached to toggle button
            container.leadingAnchor.constraint(equalTo: toggleBtn.leadingAnchor),
            container.topAnchor.constraint(equalTo: toggleBtn.bottomAnchor, constant: 8)
        ])
    }

    @objc private func toggleControls(_ sender: UIButton) {
        guard let container = controlsContainer else { return }
        
        UIView.animate(withDuration: 0.3) {
            container.isHidden.toggle()
            container.alpha = container.isHidden ? 0 : 1
        }
        
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
        case 1:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = false
        case 2:
            renderer.isMetalFXEnabled = true
            renderer.useTemporalScaler = true
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
    
    @objc private func renderScaleChanged(_ sender: UISegmentedControl) {
        let scale: Float
        let labelText: String
        switch sender.selectedSegmentIndex {
        case 0: 
            scale = 0.5
            labelText = "50%"
        case 1: 
            scale = 0.67
            labelText = "67%"
        case 2: 
            scale = 0.75
            labelText = "75%"
        case 3: 
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
}
#endif

import SwiftUI
import AVFoundation
import UIKit

public struct CameraView: View {
    @StateObject private var viewModel = ScannerViewModel()
    
    public init() {}
    
    public var body: some View {
        ZStack(alignment: .topLeading) {
            CameraPreviewView(
                session: viewModel.cameraService.session,
                overlayItems: viewModel.overlayItems
            )
            .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                Text("Frame \(viewModel.debugInfo.frameSize)")
                Text("Detections \(viewModel.debugInfo.detectionCount) | Tracks \(viewModel.debugInfo.activeTrackCount)")
                Text(viewModel.debugInfo.lastRecognitionSummary)
                    .lineLimit(2)
            }
            .font(.caption.monospacedDigit())
            .foregroundColor(.white)
            .padding(8)
            .background(Color.black.opacity(0.65))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .padding(.top, 12)
            .padding(.leading, 12)

            if let errorMessage = viewModel.errorMessage {
                VStack {
                    Spacer()
                    Text(errorMessage)
                        .font(.callout)
                        .foregroundColor(.white)
                        .padding(10)
                        .background(Color.red.opacity(0.85))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .padding()
                }
            }
        }
        .onAppear {
            if !ProcessInfo.processInfo.environment.keys.contains("XCTestConfigurationFilePath") {
                viewModel.start()
            }
        }
        .onDisappear {
            viewModel.stop()
        }
    }
}

private struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let overlayItems: [ScannerOverlayItem]
    
    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.applyPortraitRotation()
        view.overlayItems = overlayItems
        return view
    }
    
    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        uiView.previewLayer.session = session
        uiView.applyPortraitRotation()
        uiView.overlayItems = overlayItems
    }
}

private final class PreviewContainerView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    var overlayItems: [ScannerOverlayItem] = [] {
        didSet {
            drawOverlay()
        }
    }

    private let overlayLayer = CALayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        layer.addSublayer(overlayLayer)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        layer.addSublayer(overlayLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
        applyPortraitRotation()
        overlayLayer.frame = bounds
        drawOverlay()
    }

    func applyPortraitRotation() {
        guard let connection = previewLayer.connection else {
            return
        }

        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        } else if connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
    }

    private func drawOverlay() {
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }

        for item in overlayItems {
            let previewLayerRect = previewLayer.layerRectConverted(fromMetadataOutputRect: item.metadataOutputRect)
            guard previewLayerRect.width > 2, previewLayerRect.height > 2 else {
                continue
            }

            let boxLayer = CAShapeLayer()
            boxLayer.frame = bounds
            boxLayer.path = UIBezierPath(roundedRect: previewLayerRect, cornerRadius: 6).cgPath
            boxLayer.fillColor = UIColor.clear.cgColor
            boxLayer.strokeColor = color(for: item.confidence).cgColor
            boxLayer.lineWidth = 3
            overlayLayer.addSublayer(boxLayer)

            addLabel(for: item, above: previewLayerRect)
        }
    }

    private func addLabel(for item: ScannerOverlayItem, above rect: CGRect) {
        let text = "\(item.title)\n\(item.subtitle)"
        let maxWidth = min(max(rect.width, 160), bounds.width - 24)
        let labelHeight: CGFloat = 44
        let x = min(max(rect.minX, 12), max(bounds.width - maxWidth - 12, 12))
        let y = max(rect.minY - labelHeight - 6, 12)

        let backgroundLayer = CALayer()
        backgroundLayer.frame = CGRect(x: x, y: y, width: maxWidth, height: labelHeight)
        backgroundLayer.backgroundColor = UIColor.black.withAlphaComponent(0.75).cgColor
        backgroundLayer.cornerRadius = 6
        overlayLayer.addSublayer(backgroundLayer)

        let textLayer = CATextLayer()
        textLayer.frame = backgroundLayer.bounds.insetBy(dx: 8, dy: 5)
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.string = text
        textLayer.fontSize = 12
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.alignmentMode = .left
        textLayer.isWrapped = true
        backgroundLayer.addSublayer(textLayer)
    }

    private func color(for confidence: Float) -> UIColor {
        if confidence >= 0.85 {
            return UIColor.systemGreen
        }
        if confidence >= 0.6 {
            return UIColor.systemYellow
        }
        return UIColor.systemOrange
    }
} 
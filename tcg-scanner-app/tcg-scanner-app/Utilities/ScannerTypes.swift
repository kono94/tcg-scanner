import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

struct CameraFrame {
    let pixelBuffer: CVPixelBuffer
    let orientation: CGImagePropertyOrientation
    let timestamp: CMTime
}

struct DetectedCard {
    let label: String
    let confidence: Float
    let visionNormalizedRect: CGRect
    let metadataOutputRect: CGRect
}

struct TrackedCard {
    let id: UUID
    let label: String
    let confidence: Float
    let visionNormalizedRect: CGRect
    let metadataOutputRect: CGRect
}

struct RecognitionResult: Equatable {
    let cardID: String
    let name: String
    let confidence: Float
}

struct PriceQuote: Equatable {
    let displayPrice: String
    let source: String
}

struct ScannerOverlayItem: Identifiable, Equatable {
    let id: UUID
    let metadataOutputRect: CGRect
    let sourceFrameSize: CGSize
    let title: String
    let subtitle: String
    let confidence: Float
}

struct ScannerDebugInfo: Equatable {
    var frameSize: String = "-"
    var detectionCount: Int = 0
    var activeTrackCount: Int = 0
    var lastRecognitionSummary: String = "No recognition yet"
}

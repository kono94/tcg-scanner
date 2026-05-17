import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import Vision

final class CardDetector {
    private static let fallbackMinimumConfidence: Float = 0.4

    private let model: VNCoreMLModel
    private let minimumConfidence: Float

    init(minimumConfidence: Float? = nil) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try card_detector(configuration: config)
        model = try VNCoreMLModel(for: coreMLModel.model)
        self.minimumConfidence = minimumConfidence ?? Self.modelMetadataFloat(
            "detector_min_confidence",
            from: coreMLModel.model,
            fallback: Self.fallbackMinimumConfidence
        )
    }

    func detect(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) throws -> [DetectedCard] {
        var detectedCards: [DetectedCard] = []

        let request = VNCoreMLRequest(model: model) { request, _ in
            let observations = request.results as? [VNRecognizedObjectObservation] ?? []
            detectedCards = observations.compactMap { observation in
                guard let label = observation.labels.first, label.confidence >= self.minimumConfidence else {
                    return nil
                }
                let visionRect = observation.boundingBox
                return DetectedCard(
                    label: label.identifier,
                    confidence: label.confidence,
                    visionNormalizedRect: visionRect,
                    metadataOutputRect: DetectionCoordinateMapper.metadataOutputRect(fromVisionNormalizedRect: visionRect)
                )
            }
        }
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
        try handler.perform([request])
        return detectedCards
    }

    private static func modelMetadataFloat(_ key: String, from model: MLModel, fallback: Float) -> Float {
        guard let creatorDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
              let rawValue = creatorDefined[key],
              let value = Float(rawValue) else {
            return fallback
        }
        return value
    }
}

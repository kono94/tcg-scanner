import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import Vision

final class CardDetector {
    private let model: VNCoreMLModel
    private let minimumConfidence: Float

    init(minimumConfidence: Float = 0.4) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try card_detector(configuration: config)
        model = try VNCoreMLModel(for: coreMLModel.model)
        self.minimumConfidence = minimumConfidence
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
}

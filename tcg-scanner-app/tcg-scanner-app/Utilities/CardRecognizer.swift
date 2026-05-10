import CoreGraphics
import CoreML
import Foundation
import Vision

protocol CardRecognizing {
    var failureDescription: String? { get }

    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void)
}

extension CardRecognizing {
    var failureDescription: String? { nil }
}

final class CoreMLCardRecognizer: CardRecognizing {
    private struct RecognizerLabel: Decodable {
        let classIndex: Int
        let cardID: String
        let name: String
    }

    private let model: VNCoreMLModel
    private let labelsByIndex: [Int: RecognizerLabel]
    private let minimumConfidence: Float
    private let recognitionQueue = DispatchQueue(label: "net.lwenstrom.tcg-scanner.recognizer")

    init(bundle: Bundle = .main, minimumConfidence: Float = 0.5) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try card_recognizer(configuration: config)
        model = try VNCoreMLModel(for: coreMLModel.model)
        labelsByIndex = try Self.loadLabels(from: bundle)
        self.minimumConfidence = minimumConfidence
    }

    static func makeDefault() -> CardRecognizing {
        do {
            return try CoreMLCardRecognizer()
        } catch {
            return UnavailableCardRecognizer(error: error)
        }
    }

    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void) {
        recognitionQueue.async { [model, labelsByIndex, minimumConfidence] in
            var recognitionResult: RecognitionResult?
            let request = VNCoreMLRequest(model: model) { request, _ in
                guard let logits = Self.logits(from: request.results),
                      let best = Self.bestSoftmaxCandidate(from: logits),
                      best.confidence >= minimumConfidence,
                      let label = labelsByIndex[best.index] else {
                    return
                }

                recognitionResult = RecognitionResult(
                    cardID: label.cardID,
                    name: label.name,
                    confidence: best.confidence
                )
            }
            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: crop, options: [:])
            do {
                try handler.perform([request])
                completion(recognitionResult)
            } catch {
                completion(nil)
            }
        }
    }

    private static func loadLabels(from bundle: Bundle) throws -> [Int: RecognizerLabel] {
        guard let url = bundle.url(forResource: "recognizer_labels", withExtension: "json") else {
            throw RecognizerError.missingLabels
        }

        let data = try Data(contentsOf: url)
        let labels = try JSONDecoder().decode([RecognizerLabel].self, from: data)
        return Dictionary(uniqueKeysWithValues: labels.map { ($0.classIndex, $0) })
    }

    private static func logits(from results: [Any]?) -> [Float]? {
        guard let observation = results?.compactMap({ $0 as? VNCoreMLFeatureValueObservation }).first,
              let multiArray = observation.featureValue.multiArrayValue else {
            return nil
        }

        return (0..<multiArray.count).map { multiArray[$0].floatValue }
    }

    private static func bestSoftmaxCandidate(from logits: [Float]) -> (index: Int, confidence: Float)? {
        guard let maximum = logits.max() else {
            return nil
        }

        var bestIndex = 0
        var bestExp: Float = 0
        var expSum: Float = 0

        for (index, value) in logits.enumerated() {
            let expValue = expf(value - maximum)
            expSum += expValue
            if expValue > bestExp {
                bestExp = expValue
                bestIndex = index
            }
        }

        guard expSum > 0 else {
            return nil
        }
        return (bestIndex, bestExp / expSum)
    }
}

private enum RecognizerError: LocalizedError {
    case missingLabels

    var errorDescription: String? {
        switch self {
        case .missingLabels:
            return "recognizer_labels.json is missing from the app bundle."
        }
    }
}

private final class UnavailableCardRecognizer: CardRecognizing {
    let failureDescription: String?

    init(error: Error) {
        failureDescription = "The card recognizer model could not be loaded: \(error.localizedDescription)"
    }

    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void) {
        completion(nil)
    }
}

final class StubCardRecognizer: CardRecognizing {
    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void) {
        completion(nil)
    }
}

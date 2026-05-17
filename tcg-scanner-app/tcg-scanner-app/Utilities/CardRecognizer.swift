import CoreGraphics
import CoreML
import Foundation

protocol CardRecognizing {
    var failureDescription: String? { get }

    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void)
}

extension CardRecognizing {
    var failureDescription: String? { nil }
}

struct RecognizerCandidate: Equatable {
    let index: Int
    let confidence: Float
    let margin: Float
}

enum RecognizerScoring {
    static func bestCandidate(
        from logits: [Float],
        minimumConfidence: Float,
        minimumMargin: Float = 0
    ) -> RecognizerCandidate? {
        guard let maximum = logits.max() else {
            return nil
        }

        var bestIndex = 0
        var bestExp: Float = 0
        var secondBestExp: Float = 0
        var expSum: Float = 0

        for (index, value) in logits.enumerated() {
            let expValue = expf(value - maximum)
            expSum += expValue
            if expValue > bestExp {
                secondBestExp = bestExp
                bestExp = expValue
                bestIndex = index
            } else if expValue > secondBestExp {
                secondBestExp = expValue
            }
        }

        guard expSum > 0 else {
            return nil
        }

        let confidence = bestExp / expSum
        let margin = (bestExp - secondBestExp) / expSum
        guard confidence >= minimumConfidence, margin >= minimumMargin else {
            return nil
        }

        return RecognizerCandidate(index: bestIndex, confidence: confidence, margin: margin)
    }

    static func logits(from multiArray: MLMultiArray) -> [Float] {
        (0..<multiArray.count).map { multiArray[$0].floatValue }
    }
}

final class CoreMLCardRecognizer: CardRecognizing {
    private static let fallbackMinimumConfidence: Float = 0
    private static let fallbackMinimumMargin: Float = 0

    private struct RecognizerLabel: Decodable {
        let classIndex: Int
        let cardID: String
        let name: String
    }

    private let model: card_recognizer
    private let labelsByIndex: [Int: RecognizerLabel]
    private let thresholdProvider: () -> RecognitionThresholds
    private let recognitionQueue = DispatchQueue(label: "net.lwenstrom.tcg-scanner.recognizer")

    init(
        bundle: Bundle = .main,
        minimumConfidence: Float? = nil,
        minimumMargin: Float? = nil,
        thresholdProvider: (() -> RecognitionThresholds)? = nil
    ) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try card_recognizer(configuration: config)
        labelsByIndex = try Self.loadLabels(from: bundle)
        if let thresholdProvider {
            self.thresholdProvider = thresholdProvider
        } else {
            let metadataDefaults = Self.defaultThresholds(from: model.model)
            let thresholds = RecognitionThresholds(
                minimumConfidence: minimumConfidence ?? metadataDefaults.minimumConfidence,
                minimumMargin: minimumMargin ?? metadataDefaults.minimumMargin
            )
            self.thresholdProvider = { thresholds }
        }
    }

    static func makeDefault(thresholdProvider: (() -> RecognitionThresholds)? = nil) -> CardRecognizing {
        do {
            return try CoreMLCardRecognizer(thresholdProvider: thresholdProvider)
        } catch {
            return UnavailableCardRecognizer(error: error)
        }
    }

    static func defaultThresholds(bundle: Bundle = .main) -> RecognitionThresholds {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try card_recognizer(configuration: config)
            return defaultThresholds(from: model.model)
        } catch {
            return .fallback
        }
    }

    private static func defaultThresholds(from model: MLModel) -> RecognitionThresholds {
        RecognitionThresholds(
            minimumConfidence: modelMetadataFloat(
                "recognizer_min_confidence",
                from: model,
                fallback: fallbackMinimumConfidence
            ),
            minimumMargin: modelMetadataFloat(
                "recognizer_min_margin",
                from: model,
                fallback: fallbackMinimumMargin
            )
        )
    }

    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void) {
        recognitionQueue.async { [model, labelsByIndex, thresholdProvider] in
            do {
                let input = try card_recognizerInput(imageWith: crop)
                let output = try model.prediction(input: input)
                let logits = RecognizerScoring.logits(from: output.logits)
                let thresholds = thresholdProvider()
                guard let best = RecognizerScoring.bestCandidate(
                    from: logits,
                    minimumConfidence: thresholds.minimumConfidence,
                    minimumMargin: thresholds.minimumMargin
                ), let label = labelsByIndex[best.index] else {
                    completion(nil)
                    return
                }

                let recognitionResult = RecognitionResult(
                    cardID: label.cardID,
                    name: label.name,
                    confidence: best.confidence
                )
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

    private static func modelMetadataFloat(_ key: String, from model: MLModel, fallback: Float) -> Float {
        guard let creatorDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
              let rawValue = creatorDefined[key],
              let value = Float(rawValue) else {
            return fallback
        }
        return value
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

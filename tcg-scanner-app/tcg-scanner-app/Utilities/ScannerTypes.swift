import AVFoundation
import Combine
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

enum PriceCurrency: String, CaseIterable, Identifiable {
    case usd
    case eur

    var id: String { rawValue }

    var currencyCode: String {
        switch self {
        case .usd:
            return "USD"
        case .eur:
            return "EUR"
        }
    }
}

final class ScannerSettings: ObservableObject {
    @Published var useEuroPrices: Bool {
        didSet {
            userDefaults.set(useEuroPrices, forKey: Self.useEuroPricesKey)
        }
    }

    @Published var allowDuplicateCards: Bool {
        didSet {
            userDefaults.set(allowDuplicateCards, forKey: Self.allowDuplicateCardsKey)
        }
    }

    var priceCurrency: PriceCurrency {
        useEuroPrices ? .eur : .usd
    }

    private static let useEuroPricesKey = "useEuroPrices"
    private static let allowDuplicateCardsKey = "allowDuplicateCards"
    private let userDefaults: UserDefaults

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
        useEuroPrices = userDefaults.bool(forKey: Self.useEuroPricesKey)
        allowDuplicateCards = userDefaults.bool(forKey: Self.allowDuplicateCardsKey)
    }
}

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

    var cardGame: CardGame {
        CardGame(detectionLabel: label)
    }
}

struct RecognitionResult: Equatable {
    let cardID: String
    let name: String
    let confidence: Float
}

struct PriceQuote: Codable, Equatable {
    let amountUSD: Double
    let source: String

    var sortValueUSD: Double {
        amountUSD
    }

    func displayPrice(currency: PriceCurrency) -> String {
        PriceFormatter.displayPrice(usdAmount: amountUSD, currency: currency)
    }
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
    var detectorLatencyMS: Double?
    var recognizerLatencyMS: Double?
    var lastRecognitionSummary: String = "No recognition yet"

    var detectorLatencyDisplay: String {
        Self.latencyDisplay(detectorLatencyMS)
    }

    var recognizerLatencyDisplay: String {
        Self.latencyDisplay(recognizerLatencyMS)
    }

    private static func latencyDisplay(_ milliseconds: Double?) -> String {
        guard let milliseconds else {
            return "-"
        }
        return String(format: "%.1f ms", milliseconds)
    }
}

enum CardGame: Equatable {
    case onePiece
    case pokemon
    case unsupported(String)

    init(detectionLabel: String) {
        let normalizedLabel = detectionLabel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalizedLabel {
        case "op", "onepiece", "one_piece", "one piece":
            self = .onePiece
        case "pokemon", "pokémon":
            self = .pokemon
        default:
            self = .unsupported(detectionLabel)
        }
    }

    var displayName: String {
        switch self {
        case .onePiece:
            return "One Piece"
        case .pokemon:
            return "Pokemon"
        case .unsupported(let label):
            return label.isEmpty ? "Unknown" : label.capitalized
        }
    }

    var supportsExactRecognition: Bool {
        switch self {
        case .onePiece:
            return true
        case .pokemon, .unsupported:
            return false
        }
    }
}

struct SessionCard: Codable, Identifiable, Equatable {
    let id: UUID
    let trackID: UUID
    let cardID: String
    let name: String
    let matchingScore: Float
    var price: PriceQuote?
    let recognizedAt: Date
}

enum SessionSortMode: String, CaseIterable, Identifiable {
    case cardID
    case price

    var id: String { rawValue }

    var title: String {
        switch self {
        case .cardID:
            return "ID"
        case .price:
            return "Price"
        }
    }
}

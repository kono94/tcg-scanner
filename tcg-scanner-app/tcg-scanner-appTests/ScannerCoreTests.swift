import CoreGraphics
import XCTest
@testable import tcg_scanner_app

final class DetectionCoordinateMapperTests: XCTestCase {
    func testVisionRectConvertsToMetadataTopLeftCoordinates() {
        let visionRect = CGRect(x: 0.25, y: 0.10, width: 0.50, height: 0.20)

        let metadataRect = DetectionCoordinateMapper.metadataOutputRect(fromVisionNormalizedRect: visionRect)

        XCTAssertEqual(metadataRect.origin.x, 0.25, accuracy: 0.0001)
        XCTAssertEqual(metadataRect.origin.y, 0.70, accuracy: 0.0001)
        XCTAssertEqual(metadataRect.width, 0.50, accuracy: 0.0001)
        XCTAssertEqual(metadataRect.height, 0.20, accuracy: 0.0001)
    }

    func testMetadataRectConvertsToPixelBufferCoordinates() {
        let metadataRect = CGRect(x: 0.10, y: 0.20, width: 0.30, height: 0.40)

        let pixelRect = DetectionCoordinateMapper.pixelBufferRect(
            fromMetadataOutputRect: metadataRect,
            pixelBufferSize: CGSize(width: 1000, height: 500)
        )

        XCTAssertEqual(pixelRect, CGRect(x: 100, y: 100, width: 300, height: 200))
    }

    func testTopLeftRectMapsToAspectFillPreviewCoordinates() {
        let rect = CGRect(x: 0.25, y: 0.10, width: 0.50, height: 0.20)

        let previewRect = DetectionCoordinateMapper.previewLayerRect(
            fromTopLeftNormalizedRect: rect,
            sourceFrameSize: CGSize(width: 720, height: 1280),
            previewSize: CGSize(width: 360, height: 640)
        )

        XCTAssertEqual(previewRect.origin.x, 90, accuracy: 0.0001)
        XCTAssertEqual(previewRect.origin.y, 64, accuracy: 0.0001)
        XCTAssertEqual(previewRect.width, 180, accuracy: 0.0001)
        XCTAssertEqual(previewRect.height, 128, accuracy: 0.0001)
    }

    func testTopLeftRectMapsToAspectFillPreviewCoordinatesWithHorizontalCrop() {
        let rect = CGRect(x: 0.10, y: 0.25, width: 0.20, height: 0.50)

        let previewRect = DetectionCoordinateMapper.previewLayerRect(
            fromTopLeftNormalizedRect: rect,
            sourceFrameSize: CGSize(width: 720, height: 1280),
            previewSize: CGSize(width: 390, height: 844)
        )

        XCTAssertEqual(previewRect.origin.x, 5.1, accuracy: 0.0001)
        XCTAssertEqual(previewRect.origin.y, 211, accuracy: 0.0001)
        XCTAssertEqual(previewRect.width, 94.95, accuracy: 0.0001)
        XCTAssertEqual(previewRect.height, 422, accuracy: 0.0001)
    }
}

final class RecognitionSchedulerTests: XCTestCase {
    func testUnknownTrackRetriesAfterShortInterval() {
        let policy = RecognitionTimingPolicy(unknownRetryInterval: 0.5, knownRefreshInterval: 3.0, confidentMatchThreshold: 0.85)
        var state = TrackRecognitionState()
        let start = Date(timeIntervalSince1970: 100)

        XCTAssertTrue(state.shouldAttemptRecognition(now: start, policy: policy))
        state.markAttempt(now: start)

        XCTAssertFalse(state.shouldAttemptRecognition(now: start.addingTimeInterval(0.49), policy: policy))
        state.apply(result: nil, now: start.addingTimeInterval(0.1), policy: policy)
        XCTAssertFalse(state.shouldAttemptRecognition(now: start.addingTimeInterval(0.49), policy: policy))
        XCTAssertTrue(state.shouldAttemptRecognition(now: start.addingTimeInterval(0.50), policy: policy))
    }

    func testInFlightRecognitionBlocksRetry() {
        let policy = RecognitionTimingPolicy(unknownRetryInterval: 0.5, knownRefreshInterval: 3.0, confidentMatchThreshold: 0.85)
        var state = TrackRecognitionState()
        let start = Date(timeIntervalSince1970: 150)

        state.markAttempt(now: start)

        XCTAssertFalse(state.shouldAttemptRecognition(now: start.addingTimeInterval(10), policy: policy))
        state.apply(result: nil, now: start.addingTimeInterval(0.1), policy: policy)
        XCTAssertTrue(state.shouldAttemptRecognition(now: start.addingTimeInterval(10), policy: policy))
    }

    func testConfidentTrackUsesLongRefreshInterval() {
        let policy = RecognitionTimingPolicy(unknownRetryInterval: 0.5, knownRefreshInterval: 3.0, confidentMatchThreshold: 0.85)
        var state = TrackRecognitionState()
        let start = Date(timeIntervalSince1970: 200)

        state.markAttempt(now: start)
        state.apply(
            result: RecognitionResult(cardID: "OP01-001", name: "Test Card", confidence: 0.92),
            now: start,
            policy: policy
        )

        XCTAssertFalse(state.shouldAttemptRecognition(now: start.addingTimeInterval(2.99), policy: policy))
        XCTAssertTrue(state.shouldAttemptRecognition(now: start.addingTimeInterval(3.0), policy: policy))
    }

    func testTrackKeepsHighestRecognitionScore() {
        let policy = RecognitionTimingPolicy(unknownRetryInterval: 0.5, knownRefreshInterval: 3.0, confidentMatchThreshold: 0.85)
        var state = TrackRecognitionState()
        let start = Date(timeIntervalSince1970: 250)

        state.apply(
            result: RecognitionResult(cardID: "OP01-001", name: "Lower", confidence: 0.72),
            now: start,
            policy: policy
        )
        state.apply(
            result: RecognitionResult(cardID: "OP01-002", name: "Worse", confidence: 0.60),
            now: start.addingTimeInterval(1),
            policy: policy
        )
        XCTAssertEqual(state.result?.cardID, "OP01-001")
        XCTAssertEqual(state.result?.confidence, 0.72)

        state.apply(
            result: RecognitionResult(cardID: "OP01-003", name: "Better", confidence: 0.91),
            now: start.addingTimeInterval(2),
            policy: policy
        )
        XCTAssertEqual(state.result?.cardID, "OP01-003")
        XCTAssertEqual(state.result?.confidence, 0.91)
    }

    func testUnsupportedGameDoesNotAttemptExactRecognition() {
        let policy = RecognitionTimingPolicy(unknownRetryInterval: 0.5, knownRefreshInterval: 3.0, confidentMatchThreshold: 0.85)
        var state = TrackRecognitionState()
        state.markUnsupported(gameName: "Pokemon")

        XCTAssertFalse(state.shouldAttemptRecognition(now: Date(timeIntervalSince1970: 300), policy: policy))
        XCTAssertNil(state.result)
        XCTAssertTrue(state.isExactRecognitionUnsupported)
    }
}

final class RecognizerScoringTests: XCTestCase {
    func testBestCandidateAllowsClassifierResultBelowLegacyThreshold() {
        let logits: [Float] = [0.0, -0.35, -2.5, -2.5, -2.5]

        XCTAssertNil(RecognizerScoring.bestCandidate(from: logits, minimumConfidence: 0.5))

        let candidate = RecognizerScoring.bestCandidate(from: logits, minimumConfidence: 0.03)

        XCTAssertEqual(candidate?.index, 0)
        XCTAssertEqual(candidate?.confidence ?? 0, 0.4687, accuracy: 0.001)
    }

    func testBestCandidateCanRejectTinyTopTwoMargin() {
        let logits: [Float] = [0.0, -0.001, -4.0]

        XCTAssertNotNil(RecognizerScoring.bestCandidate(from: logits, minimumConfidence: 0.03, minimumMargin: 0))
        XCTAssertNil(RecognizerScoring.bestCandidate(from: logits, minimumConfidence: 0.03, minimumMargin: 0.01))
    }
}

final class CardGameTests: XCTestCase {
    func testOnePieceLabelsSupportExactRecognition() {
        XCTAssertEqual(CardGame(detectionLabel: "op"), .onePiece)
        XCTAssertTrue(CardGame(detectionLabel: "one piece").supportsExactRecognition)
    }

    func testPokemonLabelsDoNotSupportExactRecognition() {
        let game = CardGame(detectionLabel: "pokemon")

        XCTAssertEqual(game, .pokemon)
        XCTAssertEqual(game.displayName, "Pokemon")
        XCTAssertFalse(game.supportsExactRecognition)
    }
}

final class AppModelManifestTests: XCTestCase {
    func testManifestDecodesVersionFields() throws {
        let json = """
        {
          "detectorVersion": "detector-test",
          "recognizerVersion": "recognizer-test",
          "cardDBVersion": "cards-test",
          "priceSnapshotDate": "2026-05-10",
          "recognizerMinConfidence": 0.03,
          "recognizerMinMargin": 0.005,
          "recognizerThresholdSource": "test-report"
        }
        """

        let manifest = try AppModelManifest.decode(Data(json.utf8))

        XCTAssertEqual(manifest.detectorVersion, "detector-test")
        XCTAssertEqual(manifest.recognizerVersion, "recognizer-test")
        XCTAssertEqual(manifest.cardDBVersion, "cards-test")
        XCTAssertEqual(manifest.priceSnapshotDate, "2026-05-10")
        XCTAssertEqual(manifest.recognizerMinConfidence, 0.03)
        XCTAssertEqual(manifest.recognizerMinMargin, 0.005)
        XCTAssertEqual(manifest.recognizerThresholdSource, "test-report")
    }

    func testManifestDecodesWithoutRecognizerThresholdFields() throws {
        let json = """
        {
          "detectorVersion": "detector-test",
          "recognizerVersion": "recognizer-test",
          "cardDBVersion": "cards-test",
          "priceSnapshotDate": "2026-05-10"
        }
        """

        let manifest = try AppModelManifest.decode(Data(json.utf8))

        XCTAssertNil(manifest.recognizerMinConfidence)
        XCTAssertNil(manifest.recognizerMinMargin)
        XCTAssertNil(manifest.recognizerThresholdSource)
    }
}

final class ScannerDebugInfoTests: XCTestCase {
    func testLatencyDisplaysUseMilliseconds() {
        let debugInfo = ScannerDebugInfo(detectorLatencyMS: 12.345, recognizerLatencyMS: 67.89)

        XCTAssertEqual(debugInfo.detectorLatencyDisplay, "12.3 ms")
        XCTAssertEqual(debugInfo.recognizerLatencyDisplay, "67.9 ms")
    }

    func testMissingLatencyDisplaysPlaceholder() {
        let debugInfo = ScannerDebugInfo()

        XCTAssertEqual(debugInfo.detectorLatencyDisplay, "-")
        XCTAssertEqual(debugInfo.recognizerLatencyDisplay, "-")
    }
}

final class PriceFormatterTests: XCTestCase {
    func testFormatsUSD() {
        let displayPrice = PriceFormatter.displayPrice(usdAmount: 12.5, currency: .usd)

        XCTAssertTrue(displayPrice.contains("12.50"))
    }

    func testFormatsEURFromBundledUSDSnapshotValue() {
        let displayPrice = PriceFormatter.displayPrice(usdAmount: 10, currency: .eur)

        XCTAssertTrue(displayPrice.contains("9.20"))
    }
}

final class ScannerSettingsTests: XCTestCase {
    func testDuplicateCardsAreDisabledByDefault() {
        let suiteName = "ScannerSettingsTests.\(UUID().uuidString)"
        let userDefaults = UserDefaults(suiteName: suiteName)!
        defer { userDefaults.removePersistentDomain(forName: suiteName) }

        let settings = ScannerSettings(userDefaults: userDefaults)

        XCTAssertFalse(settings.allowDuplicateCards)
    }

    func testDuplicateCardsSettingPersists() {
        let suiteName = "ScannerSettingsTests.\(UUID().uuidString)"
        let userDefaults = UserDefaults(suiteName: suiteName)!
        defer { userDefaults.removePersistentDomain(forName: suiteName) }

        let settings = ScannerSettings(userDefaults: userDefaults)
        settings.allowDuplicateCards = true

        XCTAssertTrue(ScannerSettings(userDefaults: userDefaults).allowDuplicateCards)
    }

    func testRecognitionThresholdsUseProvidedDefaults() {
        let suiteName = "ScannerSettingsTests.\(UUID().uuidString)"
        let userDefaults = UserDefaults(suiteName: suiteName)!
        defer { userDefaults.removePersistentDomain(forName: suiteName) }
        let defaults = RecognitionThresholds(minimumConfidence: 0.4, minimumMargin: 0.01)

        let settings = ScannerSettings(userDefaults: userDefaults, defaultRecognitionThresholds: defaults)

        XCTAssertEqual(settings.recognizerMinimumConfidence, 0.4)
        XCTAssertEqual(settings.recognizerMinimumMargin, 0.01)
        XCTAssertEqual(settings.recognitionThresholds, defaults)
    }

    func testRecognitionThresholdsPersist() {
        let suiteName = "ScannerSettingsTests.\(UUID().uuidString)"
        let userDefaults = UserDefaults(suiteName: suiteName)!
        defer { userDefaults.removePersistentDomain(forName: suiteName) }
        let defaults = RecognitionThresholds(minimumConfidence: 0.4, minimumMargin: 0.01)

        let settings = ScannerSettings(userDefaults: userDefaults, defaultRecognitionThresholds: defaults)
        settings.recognizerMinimumConfidence = 0.25
        settings.recognizerMinimumMargin = 0.05

        let reloaded = ScannerSettings(userDefaults: userDefaults, defaultRecognitionThresholds: defaults)
        XCTAssertEqual(reloaded.recognizerMinimumConfidence, 0.25)
        XCTAssertEqual(reloaded.recognizerMinimumMargin, 0.05)
    }

    func testResetRecognitionThresholdsRestoresDefaults() {
        let suiteName = "ScannerSettingsTests.\(UUID().uuidString)"
        let userDefaults = UserDefaults(suiteName: suiteName)!
        defer { userDefaults.removePersistentDomain(forName: suiteName) }
        let defaults = RecognitionThresholds(minimumConfidence: 0.4, minimumMargin: 0.01)
        let settings = ScannerSettings(userDefaults: userDefaults, defaultRecognitionThresholds: defaults)
        settings.recognizerMinimumConfidence = 0.25
        settings.recognizerMinimumMargin = 0.05

        settings.resetRecognitionThresholdsToDefaults()

        XCTAssertEqual(settings.recognizerMinimumConfidence, 0.4)
        XCTAssertEqual(settings.recognizerMinimumMargin, 0.01)
        let reloaded = ScannerSettings(userDefaults: userDefaults, defaultRecognitionThresholds: defaults)
        XCTAssertEqual(reloaded.recognitionThresholds, defaults)
    }
}

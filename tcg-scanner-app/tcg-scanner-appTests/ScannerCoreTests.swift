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
}

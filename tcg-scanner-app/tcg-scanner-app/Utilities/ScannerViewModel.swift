import AVFoundation
import Combine
import CoreGraphics
import CoreVideo
import Foundation

final class ScannerViewModel: ObservableObject {
    @Published private(set) var overlayItems: [ScannerOverlayItem] = []
    @Published private(set) var debugInfo = ScannerDebugInfo()
    @Published private(set) var errorMessage: String?

    let cameraService: CameraService

    private var detector: CardDetector?
    private var didAttemptDetectorLoad = false
    private let tracker = CardTracker()
    private let cropper = CardCropper()
    private let recognizer: CardRecognizing
    private let metadataStore: CardMetadataStore
    private let priceService: PriceServing
    private let scannerQueue = DispatchQueue(label: "net.lwenstrom.tcg-scanner.scanner")
    private let recognitionPolicy = RecognitionTimingPolicy()
    private var recognitionStates: [UUID: TrackRecognitionState] = [:]
    private var lastDetectionTime: Date = .distantPast
    private var isProcessingFrame = false
    private let detectionInterval: TimeInterval = 0.2

    init(
        cameraService: CameraService = CameraService(),
        recognizer: CardRecognizing = StubCardRecognizer(),
        metadataStore: CardMetadataStore = CardMetadataStore(),
        priceService: PriceServing = MockPriceService()
    ) {
        self.cameraService = cameraService
        self.recognizer = recognizer
        self.metadataStore = metadataStore
        self.priceService = priceService
    }

    func start() {
        cameraService.frameHandler = { [weak self] frame in
            self?.handle(frame: frame)
        }
        cameraService.start()
    }

    func stop() {
        cameraService.stop()
        cameraService.frameHandler = nil
        scannerQueue.async { [weak self] in
            self?.tracker.reset()
            self?.recognitionStates.removeAll()
        }
    }

    private func handle(frame: CameraFrame) {
        scannerQueue.async { [weak self] in
            guard let self, !self.isProcessingFrame else {
                return
            }
            self.isProcessingFrame = true
            defer { self.isProcessingFrame = false }

            let now = Date()
            let detections = self.runDetectionIfNeeded(frame: frame, now: now)
            let tracks = self.tracker.update(
                pixelBuffer: frame.pixelBuffer,
                orientation: frame.orientation,
                detections: detections
            )

            self.scheduleRecognitionIfNeeded(for: tracks, frame: frame, now: now)
            self.publish(tracks: tracks, detections: detections ?? [], frame: frame)
        }
    }

    private func runDetectionIfNeeded(frame: CameraFrame, now: Date) -> [DetectedCard]? {
        guard now.timeIntervalSince(lastDetectionTime) >= detectionInterval else {
            return nil
        }

        lastDetectionTime = now
        guard let detector = loadDetectorIfNeeded() else {
            return []
        }

        do {
            return try detector.detect(pixelBuffer: frame.pixelBuffer, orientation: frame.orientation)
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Detection failed: \(error.localizedDescription)"
            }
            return []
        }
    }

    private func loadDetectorIfNeeded() -> CardDetector? {
        if let detector {
            return detector
        }
        guard !didAttemptDetectorLoad else {
            return nil
        }

        didAttemptDetectorLoad = true
        do {
            let detector = try CardDetector()
            self.detector = detector
            return detector
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "The card detector model could not be loaded: \(error.localizedDescription)"
            }
            return nil
        }
    }

    private func scheduleRecognitionIfNeeded(for tracks: [TrackedCard], frame: CameraFrame, now: Date) {
        let activeIDs = Set(tracks.map(\.id))
        recognitionStates = recognitionStates.filter { activeIDs.contains($0.key) }

        for track in tracks {
            var state = recognitionStates[track.id] ?? TrackRecognitionState()
            guard state.shouldAttemptRecognition(now: now, policy: recognitionPolicy) else {
                recognitionStates[track.id] = state
                continue
            }

            state.markAttempt(now: now)
            recognitionStates[track.id] = state

            guard let crop = cropper.crop(pixelBuffer: frame.pixelBuffer, metadataOutputRect: track.metadataOutputRect) else {
                continue
            }

            recognizer.recognize(crop: crop) { [weak self] result in
                self?.scannerQueue.async {
                    guard let self else { return }
                    var updatedState = self.recognitionStates[track.id] ?? TrackRecognitionState()
                    let enrichedResult = self.enrich(result: result)
                    updatedState.apply(result: enrichedResult, now: Date(), policy: self.recognitionPolicy)
                    self.recognitionStates[track.id] = updatedState

                    if let cardID = enrichedResult?.cardID {
                        self.priceService.price(for: cardID) { [weak self] quote in
                            self?.scannerQueue.async {
                                self?.recognitionStates[track.id]?.price = quote
                            }
                        }
                    }
                }
            }
        }
    }

    private func enrich(result: RecognitionResult?) -> RecognitionResult? {
        guard let result else {
            return nil
        }
        guard let metadata = metadataStore.metadata(for: result.cardID), metadata.name != result.name else {
            return result
        }
        return RecognitionResult(cardID: result.cardID, name: metadata.name, confidence: result.confidence)
    }

    private func publish(tracks: [TrackedCard], detections: [DetectedCard], frame: CameraFrame) {
        let sourceFrameSize = CGSize(
            width: CVPixelBufferGetWidth(frame.pixelBuffer),
            height: CVPixelBufferGetHeight(frame.pixelBuffer)
        )
        let items = tracks.map { track -> ScannerOverlayItem in
            let recognition = recognitionStates[track.id]?.result
            let price = recognitionStates[track.id]?.price
            let title = recognition?.name ?? "Tracking \(track.label)"
            let confidence = recognition?.confidence ?? track.confidence
            let subtitleParts = [
                recognition?.cardID,
                price?.displayPrice ?? "Price pending",
                "\(Int(confidence * 100))%"
            ].compactMap { $0 }

            return ScannerOverlayItem(
                id: track.id,
                metadataOutputRect: track.metadataOutputRect,
                sourceFrameSize: sourceFrameSize,
                title: title,
                subtitle: subtitleParts.joined(separator: " | "),
                confidence: confidence
            )
        }

        let frameSize = "\(Int(sourceFrameSize.width))x\(Int(sourceFrameSize.height))"
        let recognitionSummary = recognitionStates.values
            .compactMap { $0.result?.cardID }
            .sorted()
            .joined(separator: ", ")

        DispatchQueue.main.async {
            self.overlayItems = items
            self.debugInfo = ScannerDebugInfo(
                frameSize: frameSize,
                detectionCount: detections.count,
                activeTrackCount: tracks.count,
                lastRecognitionSummary: recognitionSummary.isEmpty ? "No recognition yet" : recognitionSummary
            )
            self.errorMessage = self.errorMessage ?? self.cameraService.errorMessage
        }
    }
}

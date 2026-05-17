import AVFoundation
import Combine
import CoreGraphics
import CoreVideo
import Foundation

final class ScannerViewModel: ObservableObject {
    @Published private(set) var overlayItems: [ScannerOverlayItem] = []
    @Published private(set) var sessionCards: [SessionCard] = []
    @Published private(set) var debugInfo = ScannerDebugInfo()
    @Published private(set) var errorMessage: String?
    @Published private(set) var isCameraReady = false

    let cameraService: CameraService
    let settings: ScannerSettings

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
    private var sessionCardsStorage: [SessionCard] = []
    private var lastDetectionTime: Date = .distantPast
    private var latestDetectorLatencyMS: Double?
    private var latestRecognizerLatencyMS: Double?
    private var isProcessingFrame = false
    private let detectionInterval: TimeInterval = 0.2
    private let sessionStorageKey = "scannerSessionCards"
    private let userDefaults: UserDefaults

    init(
        cameraService: CameraService = CameraService(),
        settings: ScannerSettings = ScannerSettings(),
        recognizer: CardRecognizing? = nil,
        metadataStore: CardMetadataStore = CardMetadataStore(),
        priceService: PriceServing = BundledPriceService(),
        userDefaults: UserDefaults = .standard
    ) {
        self.cameraService = cameraService
        self.settings = settings
        let activeRecognizer = recognizer ?? CoreMLCardRecognizer.makeDefault(
            thresholdProvider: { [weak settings] in
                settings?.recognitionThresholds ?? .fallback
            }
        )
        self.recognizer = activeRecognizer
        self.metadataStore = metadataStore
        self.priceService = priceService
        self.userDefaults = userDefaults
        self.sessionCardsStorage = Self.loadSessionCards(from: userDefaults, key: sessionStorageKey)
        self.sessionCards = sessionCardsStorage
        self.errorMessage = activeRecognizer.failureDescription
    }

    func start() {
        DispatchQueue.main.async {
            self.isCameraReady = false
        }
        cameraService.frameHandler = { [weak self] frame in
            self?.handle(frame: frame)
        }
        cameraService.start()
    }

    func stop() {
        cameraService.stop()
        cameraService.frameHandler = nil
        DispatchQueue.main.async {
            self.isCameraReady = false
        }
        scannerQueue.async { [weak self] in
            guard let self else { return }
            self.finalizeSessionCards(forLostTrackIDs: Set(self.recognitionStates.keys))
            self.tracker.reset()
            self.recognitionStates.removeAll()
        }
    }

    func resetSession() {
        scannerQueue.async { [weak self] in
            guard let self else { return }
            self.sessionCardsStorage.removeAll()
            self.persistSessionCards()
            DispatchQueue.main.async {
                self.sessionCards = []
            }
        }
    }

    func deleteSessionCards(ids: Set<UUID>) {
        scannerQueue.async { [weak self] in
            guard let self else { return }
            self.sessionCardsStorage.removeAll { ids.contains($0.id) }
            self.persistSessionCards()
            self.publishSessionCards()
        }
    }

    func deleteSessionCards(cardIDs: Set<String>) {
        scannerQueue.async { [weak self] in
            guard let self else { return }
            self.sessionCardsStorage.removeAll { cardIDs.contains($0.cardID) }
            self.persistSessionCards()
            self.publishSessionCards()
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

        let startTime = DispatchTime.now()
        do {
            let detectedCards = try detector.detect(pixelBuffer: frame.pixelBuffer, orientation: frame.orientation)
            recordDetectorLatency(since: startTime)
            return detectedCards
        } catch {
            recordDetectorLatency(since: startTime)
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
        let lostTrackIDs = Set(recognitionStates.keys).subtracting(activeIDs)
        finalizeSessionCards(forLostTrackIDs: lostTrackIDs)
        recognitionStates = recognitionStates.filter { activeIDs.contains($0.key) }

        for track in tracks {
            var state = recognitionStates[track.id] ?? TrackRecognitionState()
            guard track.cardGame.supportsExactRecognition else {
                state.markUnsupported(gameName: track.cardGame.displayName)
                recognitionStates[track.id] = state
                continue
            }

            state.clearUnsupported()
            guard state.shouldAttemptRecognition(now: now, policy: recognitionPolicy) else {
                recognitionStates[track.id] = state
                continue
            }

            state.markAttempt(now: now)
            recognitionStates[track.id] = state

            guard let crop = cropper.crop(pixelBuffer: frame.pixelBuffer, metadataOutputRect: track.metadataOutputRect) else {
                state.isRecognitionInFlight = false
                recognitionStates[track.id] = state
                continue
            }

            let recognitionStartTime = DispatchTime.now()
            recognizer.recognize(crop: crop) { [weak self] result in
                let recognitionLatencyMS = Self.elapsedMilliseconds(since: recognitionStartTime)
                self?.scannerQueue.async {
                    guard let self else { return }
                    self.recordRecognizerLatency(recognitionLatencyMS)
                    var updatedState = self.recognitionStates[track.id] ?? TrackRecognitionState()
                    let enrichedResult = self.enrich(result: result)
                    updatedState.apply(result: enrichedResult, now: Date(), policy: self.recognitionPolicy)
                    self.recognitionStates[track.id] = updatedState

                    if let cardID = updatedState.result?.cardID, updatedState.price == nil {
                        self.priceService.price(for: cardID) { [weak self] quote in
                            self?.scannerQueue.async {
                                guard self?.recognitionStates[track.id]?.result?.cardID == cardID else {
                                    return
                                }
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

    private func finalizeSessionCards(forLostTrackIDs lostTrackIDs: Set<UUID>) {
        guard !lostTrackIDs.isEmpty else {
            return
        }

        var didChange = false
        for trackID in lostTrackIDs {
            guard let state = recognitionStates[trackID],
                  let result = state.result,
                  !sessionCardsStorage.contains(where: { $0.trackID == trackID }) else {
                continue
            }
            guard settings.allowDuplicateCards || !sessionCardsStorage.contains(where: { $0.cardID == result.cardID }) else {
                continue
            }

            let sessionCard = SessionCard(
                id: UUID(),
                trackID: trackID,
                cardID: result.cardID,
                name: result.name,
                matchingScore: result.confidence,
                price: state.price,
                recognizedAt: Date()
            )
            sessionCardsStorage.append(sessionCard)
            didChange = true

            if state.price == nil {
                priceService.price(for: result.cardID) { [weak self] quote in
                    self?.scannerQueue.async {
                        guard let self,
                              let index = self.sessionCardsStorage.firstIndex(where: { $0.trackID == trackID }) else {
                            return
                        }
                        self.sessionCardsStorage[index].price = quote
                        self.persistSessionCards()
                        self.publishSessionCards()
                    }
                }
            }
        }

        if didChange {
            persistSessionCards()
            publishSessionCards()
        }
    }

    private func publishSessionCards() {
        let cards = sessionCardsStorage
        DispatchQueue.main.async {
            self.sessionCards = cards
        }
    }

    private func persistSessionCards() {
        guard let data = try? JSONEncoder().encode(sessionCardsStorage) else {
            return
        }
        userDefaults.set(data, forKey: sessionStorageKey)
    }

    private static func loadSessionCards(from userDefaults: UserDefaults, key: String) -> [SessionCard] {
        guard let data = userDefaults.data(forKey: key),
              let cards = try? JSONDecoder().decode([SessionCard].self, from: data) else {
            return []
        }
        return cards
    }

    private func publish(tracks: [TrackedCard], detections: [DetectedCard], frame: CameraFrame) {
        let sourceFrameSize = CGSize(
            width: CVPixelBufferGetWidth(frame.pixelBuffer),
            height: CVPixelBufferGetHeight(frame.pixelBuffer)
        )
        let items = tracks.map { track -> ScannerOverlayItem in
            let recognitionState = recognitionStates[track.id]
            let recognition = recognitionState?.result
            let price = recognitionState?.price
            let title: String
            if let recognition {
                title = recognition.name
            } else if let unsupportedGameName = recognitionState?.unsupportedGameName {
                title = "Tracking \(unsupportedGameName)"
            } else {
                title = recognitionState?.isRecognitionInFlight == true ? "Recognizing..." : "Tracking \(track.cardGame.displayName)"
            }
            let matchingScore = recognition?.confidence
            var subtitleParts = [String]()
            if recognitionState?.isExactRecognitionUnsupported == true {
                subtitleParts.append("Unsupported")
            }
            if let cardID = recognition?.cardID {
                subtitleParts.append(cardID)
            }
            if let displayPrice = price?.displayPrice(currency: settings.priceCurrency) {
                subtitleParts.append(displayPrice)
            }
            if let matchingScore {
                subtitleParts.append("Match \(Int(matchingScore * 100))%")
            }

            return ScannerOverlayItem(
                id: track.id,
                metadataOutputRect: track.metadataOutputRect,
                sourceFrameSize: sourceFrameSize,
                title: title,
                subtitle: subtitleParts.joined(separator: " | "),
                confidence: matchingScore ?? 0
            )
        }

        let frameSize = "\(Int(sourceFrameSize.width))x\(Int(sourceFrameSize.height))"
        let recognitionSummary = recognitionStates.values
            .compactMap { $0.result?.cardID }
            .sorted()
            .joined(separator: ", ")

        DispatchQueue.main.async {
            self.isCameraReady = true
            self.overlayItems = items
            self.debugInfo = ScannerDebugInfo(
                frameSize: frameSize,
                detectionCount: detections.count,
                activeTrackCount: tracks.count,
                detectorLatencyMS: self.latestDetectorLatencyMS,
                recognizerLatencyMS: self.latestRecognizerLatencyMS,
                lastRecognitionSummary: recognitionSummary.isEmpty ? "No recognition yet" : recognitionSummary
            )
            self.errorMessage = self.errorMessage ?? self.cameraService.errorMessage
        }
    }

    private func recordDetectorLatency(since startTime: DispatchTime) {
        let latencyMS = Self.elapsedMilliseconds(since: startTime)
        latestDetectorLatencyMS = latencyMS
        DispatchQueue.main.async {
            self.debugInfo.detectorLatencyMS = latencyMS
        }
    }

    private func recordRecognizerLatency(_ latencyMS: Double) {
        latestRecognizerLatencyMS = latencyMS
        DispatchQueue.main.async {
            self.debugInfo.recognizerLatencyMS = latencyMS
        }
    }

    private static func elapsedMilliseconds(since startTime: DispatchTime) -> Double {
        let elapsedNanoseconds = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        return Double(elapsedNanoseconds) / 1_000_000
    }
}

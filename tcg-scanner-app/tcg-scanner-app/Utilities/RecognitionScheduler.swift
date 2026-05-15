import Foundation

struct RecognitionTimingPolicy: Equatable {
    var unknownRetryInterval: TimeInterval = 0.5
    var knownRefreshInterval: TimeInterval = 3.0
    var confidentMatchThreshold: Float = 0.85
}

struct TrackRecognitionState: Equatable {
    var result: RecognitionResult?
    var price: PriceQuote?
    var lastAttempt: Date?
    var lastSuccess: Date?
    var isRecognitionInFlight = false
    var unsupportedGameName: String?

    var isExactRecognitionUnsupported: Bool {
        unsupportedGameName != nil
    }

    func shouldAttemptRecognition(now: Date, policy: RecognitionTimingPolicy) -> Bool {
        guard !isRecognitionInFlight else {
            return false
        }
        guard !isExactRecognitionUnsupported else {
            return false
        }

        let interval = hasConfidentMatch(policy: policy) ? policy.knownRefreshInterval : policy.unknownRetryInterval
        guard let lastAttempt else {
            return true
        }
        return now.timeIntervalSince(lastAttempt) >= interval
    }

    mutating func markAttempt(now: Date) {
        lastAttempt = now
        isRecognitionInFlight = true
        unsupportedGameName = nil
    }

    mutating func apply(result: RecognitionResult?, now: Date, policy: RecognitionTimingPolicy) {
        isRecognitionInFlight = false
        unsupportedGameName = nil
        guard let result else {
            return
        }
        if self.result == nil || result.confidence > self.result!.confidence {
            self.result = result
            price = nil
        }
        if hasConfidentMatch(policy: policy) {
            lastSuccess = now
        }
    }

    func hasConfidentMatch(policy: RecognitionTimingPolicy) -> Bool {
        guard let result else {
            return false
        }
        return result.confidence >= policy.confidentMatchThreshold
    }

    mutating func markUnsupported(gameName: String) {
        result = nil
        price = nil
        isRecognitionInFlight = false
        unsupportedGameName = gameName
    }

    mutating func clearUnsupported() {
        unsupportedGameName = nil
    }
}

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

    func shouldAttemptRecognition(now: Date, policy: RecognitionTimingPolicy) -> Bool {
        let interval = hasConfidentMatch(policy: policy) ? policy.knownRefreshInterval : policy.unknownRetryInterval
        guard let lastAttempt else {
            return true
        }
        return now.timeIntervalSince(lastAttempt) >= interval
    }

    mutating func markAttempt(now: Date) {
        lastAttempt = now
    }

    mutating func apply(result: RecognitionResult?, now: Date, policy: RecognitionTimingPolicy) {
        guard let result else {
            return
        }
        self.result = result
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
}

import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Vision

final class CardTracker {
    private struct Track {
        var id: UUID
        var label: String
        var confidence: Float
        var observation: VNDetectedObjectObservation
        var missedFrames: Int
    }

    private let sequenceHandler = VNSequenceRequestHandler()
    private var tracks: [UUID: Track] = [:]
    private let matchThreshold: CGFloat = 0.35
    private let maximumMissedFrames = 8

    func reset() {
        tracks.removeAll()
    }

    func update(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        detections: [DetectedCard]?
    ) -> [TrackedCard] {
        updateExistingTracks(pixelBuffer: pixelBuffer, orientation: orientation)

        if let detections {
            mergeDetections(detections)
        }

        return tracks.values
            .filter { $0.missedFrames <= maximumMissedFrames }
            .map { track in
                TrackedCard(
                    id: track.id,
                    label: track.label,
                    confidence: track.confidence,
                    visionNormalizedRect: track.observation.boundingBox,
                    metadataOutputRect: DetectionCoordinateMapper.metadataOutputRect(fromVisionNormalizedRect: track.observation.boundingBox)
                )
            }
            .sorted { $0.id.uuidString < $1.id.uuidString }
    }

    private func updateExistingTracks(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
        guard !tracks.isEmpty else {
            return
        }

        let trackIDs = Array(tracks.keys)
        let requests = trackIDs.compactMap { id -> VNTrackObjectRequest? in
            guard let track = tracks[id] else { return nil }
            let request = VNTrackObjectRequest(detectedObjectObservation: track.observation)
            request.trackingLevel = .fast
            return request
        }

        do {
            try sequenceHandler.perform(requests, on: pixelBuffer, orientation: orientation)
            for (index, request) in requests.enumerated() {
                let id = trackIDs[index]
                guard var track = tracks[id],
                      let observation = request.results?.first as? VNDetectedObjectObservation else {
                    continue
                }

                if observation.confidence > 0.2 {
                    track.observation = observation
                    track.confidence = max(track.confidence * 0.95, observation.confidence)
                    track.missedFrames = 0
                } else {
                    track.missedFrames += 1
                }
                tracks[id] = track
            }
        } catch {
            for id in trackIDs {
                tracks[id]?.missedFrames += 1
            }
        }

        tracks = tracks.filter { $0.value.missedFrames <= maximumMissedFrames }
    }

    private func mergeDetections(_ detections: [DetectedCard]) {
        for detection in detections {
            if let matchedID = bestTrackID(for: detection.visionNormalizedRect), var track = tracks[matchedID] {
                track.label = detection.label
                track.confidence = detection.confidence
                track.observation = VNDetectedObjectObservation(boundingBox: detection.visionNormalizedRect)
                track.missedFrames = 0
                tracks[matchedID] = track
            } else {
                let id = UUID()
                tracks[id] = Track(
                    id: id,
                    label: detection.label,
                    confidence: detection.confidence,
                    observation: VNDetectedObjectObservation(boundingBox: detection.visionNormalizedRect),
                    missedFrames: 0
                )
            }
        }
    }

    private func bestTrackID(for rect: CGRect) -> UUID? {
        tracks
            .map { (id: $0.key, score: iou($0.value.observation.boundingBox, rect)) }
            .filter { $0.score >= matchThreshold }
            .max { $0.score < $1.score }?
            .id
    }

    private func iou(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        let intersection = lhs.intersection(rhs)
        guard !intersection.isNull else {
            return 0
        }

        let intersectionArea = intersection.width * intersection.height
        let unionArea = lhs.width * lhs.height + rhs.width * rhs.height - intersectionArea
        guard unionArea > 0 else {
            return 0
        }
        return intersectionArea / unionArea
    }
}

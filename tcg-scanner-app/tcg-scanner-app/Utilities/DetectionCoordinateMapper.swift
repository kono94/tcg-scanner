import CoreGraphics
import Foundation

enum DetectionCoordinateMapper {
    static func metadataOutputRect(fromTopLeftNormalizedRect rect: CGRect) -> CGRect {
        rect.standardized.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    static func visionNormalizedRect(fromTopLeftNormalizedRect rect: CGRect) -> CGRect {
        let normalized = rect.standardized
        return CGRect(
            x: clamp(normalized.minX),
            y: clamp(1 - normalized.maxY),
            width: clamp(normalized.width),
            height: clamp(normalized.height)
        ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    static func metadataOutputRect(fromVisionNormalizedRect rect: CGRect) -> CGRect {
        let normalized = rect.standardized
        return CGRect(
            x: clamp(normalized.minX),
            y: clamp(1 - normalized.maxY),
            width: clamp(normalized.width),
            height: clamp(normalized.height)
        ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    static func pixelBufferRect(fromMetadataOutputRect rect: CGRect, pixelBufferSize: CGSize) -> CGRect {
        let normalized = rect.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        let pixelRect = CGRect(
            x: normalized.minX * pixelBufferSize.width,
            y: normalized.minY * pixelBufferSize.height,
            width: normalized.width * pixelBufferSize.width,
            height: normalized.height * pixelBufferSize.height
        )
        return CGRect(
            x: pixelRect.origin.x.rounded(),
            y: pixelRect.origin.y.rounded(),
            width: pixelRect.width.rounded(),
            height: pixelRect.height.rounded()
        )
    }

    private static func clamp(_ value: CGFloat) -> CGFloat {
        min(max(value, 0), 1)
    }
}

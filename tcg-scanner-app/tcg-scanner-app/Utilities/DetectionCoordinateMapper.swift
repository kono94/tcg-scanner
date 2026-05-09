import CoreGraphics
import Foundation

enum DetectionCoordinateMapper {
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
        return CGRect(
            x: normalized.minX * pixelBufferSize.width,
            y: normalized.minY * pixelBufferSize.height,
            width: normalized.width * pixelBufferSize.width,
            height: normalized.height * pixelBufferSize.height
        ).integral
    }

    private static func clamp(_ value: CGFloat) -> CGFloat {
        min(max(value, 0), 1)
    }
}

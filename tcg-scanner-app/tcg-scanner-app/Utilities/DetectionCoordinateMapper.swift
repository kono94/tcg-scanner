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

    static func previewLayerRect(
        fromTopLeftNormalizedRect rect: CGRect,
        sourceFrameSize: CGSize,
        previewSize: CGSize
    ) -> CGRect {
        guard sourceFrameSize.width > 0,
              sourceFrameSize.height > 0,
              previewSize.width > 0,
              previewSize.height > 0 else {
            return .null
        }

        let normalized = rect.standardized.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !normalized.isNull else {
            return .null
        }

        let scale = max(
            previewSize.width / sourceFrameSize.width,
            previewSize.height / sourceFrameSize.height
        )
        let displayedSize = CGSize(
            width: sourceFrameSize.width * scale,
            height: sourceFrameSize.height * scale
        )
        let offset = CGPoint(
            x: (previewSize.width - displayedSize.width) / 2,
            y: (previewSize.height - displayedSize.height) / 2
        )

        return CGRect(
            x: offset.x + normalized.minX * displayedSize.width,
            y: offset.y + normalized.minY * displayedSize.height,
            width: normalized.width * displayedSize.width,
            height: normalized.height * displayedSize.height
        )
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

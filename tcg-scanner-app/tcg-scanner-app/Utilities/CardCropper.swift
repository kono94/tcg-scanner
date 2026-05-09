import CoreGraphics
import CoreImage
import CoreVideo
import Foundation

final class CardCropper {
    private let context = CIContext()

    func crop(pixelBuffer: CVPixelBuffer, metadataOutputRect: CGRect) -> CGImage? {
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let pixelRect = DetectionCoordinateMapper.pixelBufferRect(
            fromMetadataOutputRect: metadataOutputRect,
            pixelBufferSize: CGSize(width: width, height: height)
        )

        guard pixelRect.width > 8, pixelRect.height > 8 else {
            return nil
        }

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let ciRect = CGRect(
            x: pixelRect.minX,
            y: height - pixelRect.maxY,
            width: pixelRect.width,
            height: pixelRect.height
        ).integral
        return context.createCGImage(image, from: ciRect)
    }
}

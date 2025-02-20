import CoreML
import Vision
import CoreImage

class YOLO {
    private var model: VNCoreMLModel?
    
    init() {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndGPU
            let coreMLModel = try card_detector(configuration: configuration)
            self.model = try VNCoreMLModel(for: coreMLModel.model)
            print("Model loaded successfully!")
        } catch {
            fatalError("Failed to load CoreML model: \(error)")
        }
    }
    
    func detect(image: CIImage, completion: @escaping ([Prediction]) -> Void) {
        guard let model = model else {
            completion([])
            return
        }
        
        let originalSize = image.extent.size
        let targetSize: CGFloat = 480.0 // YOLO model input size
        
        // Calculate scale and padding to maintain aspect ratio
        let scale = min(targetSize / originalSize.width, targetSize / originalSize.height)
        let scaledSize = CGSize(width: originalSize.width * scale, height: originalSize.height * scale)
        let paddingX = (targetSize - scaledSize.width) / 2
        let paddingY = (targetSize - scaledSize.height) / 2
        
        // Normalized bounds of the image content in the 480x480 input
        let xNormLeft = paddingX / targetSize
        let xNormRight = (paddingX + scaledSize.width) / targetSize
        let yNormBottom = paddingY / targetSize
        let yNormTop = (paddingY + scaledSize.height) / targetSize
        
        guard let pixelBuffer = preprocess(image: image, to: targetSize) else {
            completion([])
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation], !results.isEmpty else {
                completion([])
                return
            }
            
            let predictions = results.map { observation in
                let bb = observation.boundingBox
                
                // Adjust bounding box to original image's normalized coordinates
                let xAdjusted = (bb.origin.x - xNormLeft) / (xNormRight - xNormLeft)
                let wAdjusted = bb.width / (xNormRight - xNormLeft)
                let yAdjusted = (bb.origin.y - yNormBottom) / (yNormTop - yNormBottom)
                let hAdjusted = bb.height / (yNormTop - yNormBottom)
                
                let adjustedRect = CGRect(x: xAdjusted, y: yAdjusted, width: wAdjusted, height: hAdjusted)
                
                return Prediction(
                    label: observation.labels.first?.identifier ?? "unknown",
                    confidence: observation.confidence,
                    boundingBox: adjustedRect
                )
            }
            completion(predictions)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer).perform([request])
        } catch {
            print("Inference failed: \(error)")
            completion([])
        }
    }
    
    private func preprocess(image: CIImage, to targetSize: CGFloat) -> CVPixelBuffer? {
        let imageSize = image.extent.size
        let scale = min(targetSize / imageSize.width, targetSize / imageSize.height)
        let scaledSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let paddingX = (targetSize - scaledSize.width) / 2
        let paddingY = (targetSize - scaledSize.height) / 2
        
        // Scale and center the image
        let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let paddedImage = scaledImage.transformed(by: CGAffineTransform(translationX: paddingX, y: paddingY))
        
        // Create a 480x480 pixel buffer
        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(targetSize),
            Int(targetSize),
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        let context = CIContext()
        context.render(paddedImage, to: buffer, bounds: CGRect(x: 0, y: 0, width: targetSize, height: targetSize), colorSpace: CGColorSpaceCreateDeviceRGB())
        
        return buffer
    }
    
    struct Prediction {
        let label: String
        let confidence: Float
        let boundingBox: CGRect
    }
}
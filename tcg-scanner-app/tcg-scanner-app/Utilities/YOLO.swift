import CoreML
import Vision
import CoreImage

class YOLO {
    private var model: VNCoreMLModel?
    
    init() {
        // Load the model
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndGPU // Use CPU and GPU
            let coreMLModel = try card_detector(configuration: configuration)
            self.model = try VNCoreMLModel(for: coreMLModel.model)
            print("Model loaded successfully!")
        } catch {
            fatalError("Failed to load CoreML model: \(error)")
        }
    }
    
    func detect(image: CIImage, completion: @escaping ([Prediction]) -> Void) {
        guard let model = model,
              let pixelBuffer = preprocess(image: image)
        else {
            completion([])
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation],
                  !results.isEmpty
            else {
                completion([])
                return
            }
            
            let predictions = results.map { observation in
                Prediction(
                    label: observation.labels.first?.identifier ?? "unknown",
                    confidence: observation.confidence,
                    boundingBox: observation.boundingBox
                )
            }
            completion(predictions)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
                .perform([request])
        } catch {
            print("Inference failed: \(error)")
            completion([])
        }
    }
    
    private func preprocess(image: CIImage) -> CVPixelBuffer? {
        // Resize to 480x480 and convert to RGB pixel buffer
        let resizedImage = image.resize(to: CGSize(width: 480, height: 480))
        return resizedImage?.toRGBPixelBuffer()
    }
    
    struct Prediction {
        let label: String
        let confidence: Float
        let boundingBox: CGRect
    }
}

// MARK: - CIImage Extension for Preprocessing
extension CIImage {
    func resize(to size: CGSize) -> CIImage? {
        let scaleX = size.width / extent.width
        let scaleY = size.height / extent.height
        return transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }
    
    func toRGBPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(extent.width),
            Int(extent.height),
            kCVPixelFormatType_32BGRA, // CoreML expects BGRA
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        let context = CIContext()
        context.render(self, to: buffer)
        return buffer
    }
}

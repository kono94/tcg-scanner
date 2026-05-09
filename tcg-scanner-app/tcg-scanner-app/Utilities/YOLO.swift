// MARK: - YOLO.swift
import CoreML
import Vision
import CoreImage

public class YOLO {
    public static let shared = YOLO()
    private var model: VNCoreMLModel!
    
    private init() {
        do {
            let config = MLModelConfiguration()
            let coreMLModel = try card_detector(configuration: config)
            model = try VNCoreMLModel(for: coreMLModel.model)
        } catch {
            fatalError("Failed to load YOLO model: \(error)")
        }
    }
    
    public struct Prediction: Identifiable {
        public let id = UUID()
        public let label: String
        public let confidence: Float
        public let rect: CGRect
        
        public init(label: String, confidence: Float, rect: CGRect) {
            self.label = label
            self.confidence = confidence
            self.rect = rect
        }
    }
    
    public func detect(pixelBuffer: CVPixelBuffer, completion: @escaping ([Prediction]) -> Void) {
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                completion([])
                return
            }
            
            let predictions = results.compactMap { observation -> Prediction? in
                guard let topLabel = observation.labels.first else { return nil }
                
                // Get the bounding box coordinates
                let rect = observation.boundingBox
                
                // The model expects landscape orientation, but we're in portrait
                // We need to rotate the coordinates 90 degrees clockwise
                // This means:
                // - x becomes (1 - y)
                // - y becomes x
                // - width becomes height
                // - height becomes width
                let newRect = CGRect(
                    x: 1 - rect.origin.y - rect.height,
                    y: rect.origin.x,
                    width: rect.height,
                    height: rect.width
                )
                
                return Prediction(
                    label: topLabel.identifier,
                    confidence: topLabel.confidence,
                    rect: newRect
                )
            }
            
            DispatchQueue.main.async { completion(predictions) }
        }
        
        // Configure the request
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            do { try handler.perform([request]) }
            catch { print("❌ Detection failed: \(error)") }
        }
    }
}

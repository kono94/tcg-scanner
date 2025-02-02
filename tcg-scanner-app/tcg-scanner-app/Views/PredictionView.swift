import SwiftUI
import AVFoundation
import Vision

struct PredictionView: UIViewRepresentable {
    var predictions: [YOLO.Prediction]
    var size: CGSize
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        let layersToRemove = uiView.layer.sublayers?.filter { $0.name == "prediction" } ?? []
        layersToRemove.forEach { $0.removeFromSuperlayer() }
        
        for prediction in predictions {
            let rect = convertRect(prediction.boundingBox)
            let label = "\(prediction.label) \(String(format: "%.2f", prediction.confidence))"
            
            // Draw bounding box
            let boxLayer = CAShapeLayer()
            boxLayer.name = "prediction"
            boxLayer.path = UIBezierPath(rect: rect).cgPath
            boxLayer.strokeColor = UIColor.red.cgColor
            boxLayer.fillColor = UIColor.clear.cgColor
            boxLayer.lineWidth = 2
            uiView.layer.addSublayer(boxLayer)
            
            // Draw label
            let textLayer = CATextLayer()
            textLayer.name = "prediction"
            textLayer.string = label
            textLayer.foregroundColor = UIColor.red.cgColor
            textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
            textLayer.fontSize = 14
            textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y - 18,
                                   width: rect.width, height: 18)
            uiView.layer.addSublayer(textLayer)
        }
    }
    
    private func convertRect(_ rect: CGRect) -> CGRect {
        CGRect(
            x: rect.origin.x * size.width,
            y: (1 - rect.origin.y - rect.height) * size.height,
            width: rect.width * size.width,
            height: rect.height * size.height
        )
    }
}

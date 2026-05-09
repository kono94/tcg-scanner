// MARK: - PredictionView.swift
import SwiftUI

struct PredictionView: View {
    let predictions: [YOLO.Prediction]
    let imageSize: CGSize
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(predictions) { prediction in
                BoundingBoxView(prediction: prediction, imageSize: imageSize)
                    .frame(width: geometry.size.width, height: geometry.size.height)
            }
        }
    }
}

struct BoundingBoxView: View {
    let prediction: YOLO.Prediction
    let imageSize: CGSize
    
    private func convertRect(_ rect: CGRect) -> CGRect {
        // The rect is already in the correct orientation from the YOLO class
        // We just need to scale it to the view's dimensions
        
        // Calculate the scaling factors while maintaining aspect ratio
        let viewWidth = imageSize.width
        let viewHeight = imageSize.height
        
        // Scale to fit while maintaining aspect ratio
        let scale = min(viewWidth / imageSize.width, viewHeight / imageSize.height)
        
        // Calculate the scaled dimensions
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        
        // Calculate the offset to center the image
        let xOffset = (viewWidth - scaledWidth) / 2
        let yOffset = (viewHeight - scaledHeight) / 2
        
        // Convert YOLO normalized coordinates to view coordinates
        return CGRect(
            x: rect.origin.x * scaledWidth + xOffset,
            y: rect.origin.y * scaledHeight + yOffset,
            width: rect.width * scaledWidth,
            height: rect.height * scaledHeight
        )
    }
    
    var body: some View {
        let rect = convertRect(prediction.rect)
        
        Rectangle()
            .stroke(Color.green, lineWidth: 2)
            .frame(width: rect.width, height: rect.height)
            .position(x: rect.midX, y: rect.midY)
            .overlay(
                Text("\(prediction.label) (\(Int(prediction.confidence * 100))%)")
                    .foregroundColor(.white)
                    .padding(4)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(4)
                    .offset(x: 0, y: -rect.height/2 - 20)
            )
    }
}

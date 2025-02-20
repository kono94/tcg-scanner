import SwiftUI
import AVFoundation
import Vision

struct GuessView: View {
    @State private var predictions: [YOLO.Prediction] = []
    @State private var frameSize: CGSize = .zero
    
    var body: some View {
        ZStack {
            // Camera view
            GeometryReader { geometry in
                CameraView(predictions: $predictions)
                    .onAppear { frameSize = geometry.size }
                    .overlay(
                        PredictionView(predictions: predictions, size: frameSize)
                    )
            }
        }
        .edgesIgnoringSafeArea(.all)
    }
}

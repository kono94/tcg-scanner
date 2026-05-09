// MARK: - ContentView.swift
import SwiftUI
struct ContentView: View {
    @State private var predictions: [YOLO.Prediction] = []
    
    var body: some View {
        ZStack {
            CameraView(predictions: $predictions)
                .edgesIgnoringSafeArea(.all)
            
            PredictionView(predictions: predictions,
                         imageSize: CGSize(width: 640, height: 480))
                .edgesIgnoringSafeArea(.all)
                .overlay(
                    VStack {
                        Text("Predictions: \(predictions.count)")
                            .foregroundColor(.white)
                            .padding()
                        Spacer()
                    }
                )
        }
    }
}

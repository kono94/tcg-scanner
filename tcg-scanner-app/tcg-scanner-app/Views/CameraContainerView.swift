import SwiftUI

struct CameraContainerView: View {
    @Binding var predictions: [YOLO.Prediction]
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                CameraView(predictions: $predictions)
                    .edgesIgnoringSafeArea(.all)
                
                PredictionView(
                    predictions: predictions,
                    size: geometry.size
                )
            }
        }
    }
}
import SwiftUI
import AVFoundation
import Vision
import CoreML
import UIKit
import Combine

public struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var predictions: [YOLO.Prediction] = []
    @State private var viewSize: CGSize = .zero
    
    public init() {}
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                CameraPreviewView(session: cameraManager.session)
                    .edgesIgnoringSafeArea(.all)
                    .onAppear {
                        viewSize = geometry.size
                    }
                
                ForEach(predictions) { prediction in
                    DetectionBox(prediction: prediction, viewSize: viewSize)
                }
            }
        }
        .onAppear {
            cameraManager.start()
        }
        .onDisappear {
            cameraManager.stop()
        }
        .onReceive(cameraManager.framePublisher) { pixelBuffer in
            YOLO.shared.detect(pixelBuffer: pixelBuffer) { newPredictions in
                predictions = newPredictions
            }
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.frame = view.frame
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct DetectionBox: View {
    let prediction: YOLO.Prediction
    let viewSize: CGSize
    
    var body: some View {
        let rect = convertRect(prediction.rect, to: viewSize)
        
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
    
    private func convertRect(_ rect: CGRect, to size: CGSize) -> CGRect {
        return CGRect(
            x: rect.origin.x * size.width,
            y: rect.origin.y * size.height,
            width: rect.width * size.width,
            height: rect.height * size.height
        )
    }
} 

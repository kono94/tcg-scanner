import AVFoundation
import Combine

public class CameraManager: NSObject, ObservableObject {
    public let session = AVCaptureSession()
    public let framePublisher = PassthroughSubject<CVPixelBuffer, Never>()
    
    private var videoOutput: AVCaptureVideoDataOutput?
    
    public override init() {
        super.init()
        setupCamera()
    }
    
    private func setupCamera() {
        session.beginConfiguration()
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else { return }
        
        if session.canAddInput(videoInput) {
            session.addInput(videoInput)
        }
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        self.videoOutput = videoOutput
        
        session.commitConfiguration()
    }
    
    public func start() {
        session.startRunning()
    }
    
    public func stop() {
        session.stopRunning()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        framePublisher.send(pixelBuffer)
    }
} 
// MARK: - VideoCapture.swift
import AVFoundation
import CoreGraphics
import CoreVideo
import Combine

protocol VideoCaptureDelegate: AnyObject {
    func videoCapture(_ capture: VideoCapture, didCaptureFrame pixelBuffer: CVPixelBuffer)
}

// MARK: - VideoCapture.swift (Fixed Aspect)
class VideoCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    weak var delegate: VideoCaptureDelegate?
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    var previewLayer: AVCaptureVideoPreviewLayer!

    func setupCamera(completion: @escaping (Bool) -> Void) {
        captureSession.sessionPreset = .vga640x480
        
        // Configure camera input
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("❌ Camera setup failed")
            completion(false)
            return
        }

        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        
        // Configure video output
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video.queue"))
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }

        // Configure preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        
        // Set the orientation of the video output
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
        
        completion(true)
    }

    func start() {
        if !captureSession.isRunning {
            DispatchQueue.global().async { self.captureSession.startRunning() }
        }
    }

    func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Create a new pixel buffer with the correct orientation
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var newPixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &newPixelBuffer)
        
        if let newPixelBuffer = newPixelBuffer {
            CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            CVPixelBufferLockBaseAddress(newPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            
            let srcData = CVPixelBufferGetBaseAddress(pixelBuffer)
            let dstData = CVPixelBufferGetBaseAddress(newPixelBuffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            
            memcpy(dstData, srcData, bytesPerRow * height)
            
            CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            CVPixelBufferUnlockBaseAddress(newPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            
            delegate?.videoCapture(self, didCaptureFrame: newPixelBuffer)
        }
    }
}



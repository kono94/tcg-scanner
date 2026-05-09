import AVFoundation
import Combine
import Foundation
import ImageIO

final class CameraService: NSObject, ObservableObject {
    @Published private(set) var authorizationStatus: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
    @Published private(set) var errorMessage: String?

    let session = AVCaptureSession()
    var frameHandler: ((CameraFrame) -> Void)?

    private let sessionQueue = DispatchQueue(label: "net.lwenstrom.tcg-scanner.camera.session")
    private let videoQueue = DispatchQueue(label: "net.lwenstrom.tcg-scanner.camera.frames")
    private let videoOutput = AVCaptureVideoDataOutput()
    private var isConfigured = false

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.authorizationStatus = granted ? .authorized : .denied
                    if !granted {
                        self?.errorMessage = "Camera access is required for live scanning."
                    }
                }
                if granted {
                    self?.configureAndStart()
                }
            }
        case .denied, .restricted:
            DispatchQueue.main.async {
                self.authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
                self.errorMessage = "Camera access is required for live scanning."
            }
        @unknown default:
            DispatchQueue.main.async {
                self.errorMessage = "Unsupported camera authorization state."
            }
        }
    }

    func stop() {
        sessionQueue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
    }

    private func configureAndStart() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.isConfigured {
                self.configureSession()
            }
            if self.isConfigured && !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            publish(error: "Back camera is not available.")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                publish(error: "Cannot add camera input.")
                return
            }
            session.addInput(input)
        } catch {
            publish(error: "Camera input failed: \(error.localizedDescription)")
            return
        }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        guard session.canAddOutput(videoOutput) else {
            publish(error: "Cannot add camera frame output.")
            return
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }

        isConfigured = true
    }

    private func publish(error: String) {
        DispatchQueue.main.async {
            self.errorMessage = error
        }
    }
}

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let frame = CameraFrame(
            pixelBuffer: pixelBuffer,
            orientation: .right,
            timestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        )
        frameHandler?(frame)
    }
}

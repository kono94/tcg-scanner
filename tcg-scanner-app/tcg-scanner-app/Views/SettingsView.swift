import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: ScannerSettings
    @ObservedObject var scannerViewModel: ScannerViewModel
    let modelManifest: AppModelManifest

    init(
        settings: ScannerSettings,
        scannerViewModel: ScannerViewModel,
        modelManifest: AppModelManifest = .load()
    ) {
        self.settings = settings
        self.scannerViewModel = scannerViewModel
        self.modelManifest = modelManifest
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Prices") {
                    Toggle("Use EURO", isOn: $settings.useEuroPrices)
                    Text(settings.useEuroPrices ? "Prices display in EUR." : "Prices display in USD.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Session") {
                    Toggle("Allow duplicate cards", isOn: $settings.allowDuplicateCards)
                    Text(settings.allowDuplicateCards ? "Each recognized track can add another row." : "A card ID is added only once and counted once in the total.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Recognition") {
                    ThresholdSlider(
                        title: "Confidence",
                        value: thresholdBinding(\.recognizerMinimumConfidence),
                        defaultValue: settings.defaultRecognitionThresholds.minimumConfidence
                    )
                    ThresholdSlider(
                        title: "Margin",
                        value: thresholdBinding(\.recognizerMinimumMargin),
                        defaultValue: settings.defaultRecognitionThresholds.minimumMargin
                    )
                    Button("RESET TO DEFAULT") {
                        settings.resetRecognitionThresholdsToDefaults()
                    }
                }

                Section("Scanner Diagnostics") {
                    InfoRow(title: "Detector latency", value: scannerViewModel.debugInfo.detectorLatencyDisplay)
                    InfoRow(title: "Recognizer latency", value: scannerViewModel.debugInfo.recognizerLatencyDisplay)
                    InfoRow(title: "Frame", value: scannerViewModel.debugInfo.frameSize)
                    InfoRow(title: "Detections", value: "\(scannerViewModel.debugInfo.detectionCount)")
                    InfoRow(title: "Active tracks", value: "\(scannerViewModel.debugInfo.activeTrackCount)")
                }

                Section("Model Versions") {
                    InfoRow(title: "Detector", value: modelManifest.detectorVersion)
                    InfoRow(title: "Recognizer", value: modelManifest.recognizerVersion)
                    if let recognizerMinConfidence = modelManifest.recognizerMinConfidence {
                        InfoRow(title: "Recognizer min conf", value: String(format: "%.3f", recognizerMinConfidence))
                    }
                    if let recognizerMinMargin = modelManifest.recognizerMinMargin {
                        InfoRow(title: "Recognizer min margin", value: String(format: "%.3f", recognizerMinMargin))
                    }
                    InfoRow(title: "Card DB", value: modelManifest.cardDBVersion)
                    InfoRow(title: "Price snapshot", value: modelManifest.priceSnapshotDate)
                }
            }
            .navigationTitle("Settings")
        }
    }

    private func thresholdBinding(_ keyPath: ReferenceWritableKeyPath<ScannerSettings, Float>) -> Binding<Double> {
        Binding(
            get: { Double(settings[keyPath: keyPath]) },
            set: { settings[keyPath: keyPath] = Float($0) }
        )
    }
}

private struct InfoRow: View {
    let title: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
            Spacer(minLength: 12)
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
                .font(.footnote.monospacedDigit())
        }
    }
}

private struct ThresholdSlider: View {
    let title: String
    @Binding var value: Double
    let defaultValue: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                Spacer(minLength: 12)
                Text(String(format: "%.3f", value))
                    .foregroundStyle(.secondary)
                    .font(.footnote.monospacedDigit())
            }
            Slider(value: $value, in: 0...1, step: 0.001)
            Text("Default \(String(format: "%.3f", defaultValue))")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }
}

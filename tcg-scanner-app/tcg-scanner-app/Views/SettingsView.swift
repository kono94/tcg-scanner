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
                    InfoRow(title: "Card DB", value: modelManifest.cardDBVersion)
                    InfoRow(title: "Price snapshot", value: modelManifest.priceSnapshotDate)
                }
            }
            .navigationTitle("Settings")
        }
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

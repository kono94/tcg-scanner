import SwiftUI

@main
struct TCP_ScannerApp: App {
    var body: some Scene {
        WindowGroup {
            ScannerRootView()
        }
    }
}

private struct ScannerRootView: View {
    @StateObject private var settings: ScannerSettings
    @StateObject private var scannerViewModel: ScannerViewModel

    init() {
        let settings = ScannerSettings()
        _settings = StateObject(wrappedValue: settings)
        _scannerViewModel = StateObject(wrappedValue: ScannerViewModel(settings: settings))
    }

    var body: some View {
        TabView {
            CameraView(viewModel: scannerViewModel)
                .tabItem {
                    Label("Scan", systemImage: "camera")
                }

            SessionView(viewModel: scannerViewModel, settings: settings)
                .tabItem {
                    Label("Session", systemImage: "list.bullet.rectangle")
                }

            SettingsView(settings: settings)
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
        }
    }
}

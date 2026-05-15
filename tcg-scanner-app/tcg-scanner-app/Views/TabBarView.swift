import SwiftUI

struct TabBarView: View {
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
                .tabItem { Label("Scan", systemImage: "camera.fill") }

            SessionView(viewModel: scannerViewModel, settings: settings)
                .tabItem { Label("Session", systemImage: "list.bullet.rectangle") }

            SettingsView(settings: settings, scannerViewModel: scannerViewModel)
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        .accentColor(.blue)
        .onAppear {
            UITabBar.appearance().backgroundColor = UIColor.systemBackground
        }
    }
}

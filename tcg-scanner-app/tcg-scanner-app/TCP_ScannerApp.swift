import SwiftUI

@main
struct TCP_ScannerApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                CameraView()
                    .tabItem {
                        Label("Scan", systemImage: "camera")
                    }
                
                Text("History")
                    .tabItem {
                        Label("History", systemImage: "clock")
                    }
            }
        }
    }
}

import SwiftUI

struct TabBarView: View {
    var body: some View {
        TabView {
            HomeView()
                .tabItem { Label("Home", systemImage: "house") }
            
            CameraView()
                .tabItem { Label("Scan", systemImage: "camera.fill") }
            
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        .accentColor(.blue)
        .onAppear {
            UITabBar.appearance().backgroundColor = UIColor.systemBackground
        }
    }
}

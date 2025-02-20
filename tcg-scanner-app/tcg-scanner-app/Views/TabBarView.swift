import SwiftUI
struct TabBarView: View {
    @State private var predictions: [YOLO.Prediction] = []

    var body: some View {
        TabView {
            HomeView()
                .tabItem { Label("Home", systemImage: "house") }
            
            CameraContainerView(predictions: $predictions)
                .tabItem { Label("Scan", systemImage: "camera.fill") }
            
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        .accentColor(.blue)
        .onAppear {
            // Fix tab bar transparency issue
            UITabBar.appearance().backgroundColor = UIColor.systemBackground
        }
    }
}

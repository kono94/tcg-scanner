import SwiftUI

struct TabBarView: View {
    @State private var predictions: [YOLO.Prediction] = []
    
    var body: some View {
        TabView {
            HomeView()
                .tabItem { Label("Home", systemImage: "house") }
            
            ContentView()
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

struct HomeView: View {
    var body: some View {
        VStack {
            Text("Welcome to TCG Scanner")
                .font(.title)
            Spacer()
        }
        .padding()
    }
}

struct SettingsView: View {
    var body: some View {
        Text("Settings Placeholder")
    }
}
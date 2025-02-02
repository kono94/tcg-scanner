import SwiftUI

struct LaunchScreenView: View {
    var body: some View {
        VStack {
            Image(systemName: "camera")
                .font(.system(size: 60))
            Text("TCG Scanner")
                .font(.largeTitle)
                .padding()
        }
    }
}

import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: ScannerSettings

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
            }
            .navigationTitle("Settings")
        }
    }
}

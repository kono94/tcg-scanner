import Foundation

struct AppModelManifest: Decodable, Equatable {
    let detectorVersion: String
    let recognizerVersion: String
    let cardDBVersion: String
    let priceSnapshotDate: String

    static let unavailable = AppModelManifest(
        detectorVersion: "Unavailable",
        recognizerVersion: "Unavailable",
        cardDBVersion: "Unavailable",
        priceSnapshotDate: "Unavailable"
    )

    static func load(bundle: Bundle = .main) -> AppModelManifest {
        guard let url = bundle.url(forResource: "app_model_manifest", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let manifest = try? decode(data) else {
            return .unavailable
        }
        return manifest
    }

    static func decode(_ data: Data) throws -> AppModelManifest {
        try JSONDecoder().decode(AppModelManifest.self, from: data)
    }
}

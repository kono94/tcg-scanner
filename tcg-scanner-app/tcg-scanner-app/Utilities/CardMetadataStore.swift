import Foundation

struct CardMetadata: Decodable, Equatable {
    let id: String
    let name: String
}

final class CardMetadataStore {
    private var cardsByID: [String: CardMetadata] = [:]

    init(bundle: Bundle = .main) {
        loadIndex(from: bundle)
    }

    func metadata(for cardID: String) -> CardMetadata? {
        cardsByID[cardID]
    }

    private func loadIndex(from bundle: Bundle) {
        guard let url = bundle.url(forResource: "card_index", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let cards = try? JSONDecoder().decode([CardMetadata].self, from: data) else {
            return
        }
        cardsByID = Dictionary(uniqueKeysWithValues: cards.map { ($0.id, $0) })
    }
}

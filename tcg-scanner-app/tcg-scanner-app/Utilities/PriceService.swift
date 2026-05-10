import Foundation

protocol PriceServing {
    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void)
}

final class BundledPriceService: PriceServing {
    private struct PriceRecord: Decodable {
        let id: String
        let displayPrice: String?
        let priceSource: String?
    }

    private var pricesByID: [String: PriceQuote] = [:]

    init(bundle: Bundle = .main) {
        guard let url = bundle.url(forResource: "card_index", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let records = try? JSONDecoder().decode([PriceRecord].self, from: data) else {
            return
        }

        pricesByID = Dictionary(uniqueKeysWithValues: records.compactMap { record in
            guard let displayPrice = record.displayPrice, !displayPrice.isEmpty else {
                return nil
            }

            return (
                record.id,
                PriceQuote(displayPrice: displayPrice, source: record.priceSource ?? "snapshot")
            )
        })
    }

    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void) {
        completion(pricesByID[cardID])
    }
}

final class MockPriceService: PriceServing {
    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void) {
        completion(nil)
    }
}

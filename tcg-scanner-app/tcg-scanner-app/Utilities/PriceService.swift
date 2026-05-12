import Foundation

protocol PriceServing {
    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void)
}

enum PriceFormatter {
    private static let usdToEuroRate = 0.92

    static func displayPrice(usdAmount: Double, currency: PriceCurrency) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.currencyCode = currency.currencyCode
        formatter.maximumFractionDigits = 2
        formatter.minimumFractionDigits = 2

        let amount: Double
        switch currency {
        case .usd:
            amount = usdAmount
        case .eur:
            amount = usdAmount * usdToEuroRate
        }

        return formatter.string(from: NSNumber(value: amount)) ?? fallbackDisplayPrice(amount: amount, currency: currency)
    }

    private static func fallbackDisplayPrice(amount: Double, currency: PriceCurrency) -> String {
        switch currency {
        case .usd:
            return String(format: "$%.2f", amount)
        case .eur:
            return String(format: "EUR %.2f", amount)
        }
    }
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
            guard let amountUSD = Self.usdAmount(from: record.displayPrice) else {
                return nil
            }

            return (
                record.id,
                PriceQuote(amountUSD: amountUSD, source: record.priceSource ?? "snapshot")
            )
        })
    }

    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void) {
        completion(pricesByID[cardID])
    }

    private static func usdAmount(from displayPrice: String?) -> Double? {
        guard let displayPrice, !displayPrice.isEmpty else {
            return nil
        }

        let allowedCharacters = CharacterSet(charactersIn: "0123456789.")
        let numericString = String(displayPrice.unicodeScalars.filter { allowedCharacters.contains($0) })
        return Double(numericString)
    }
}

final class MockPriceService: PriceServing {
    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void) {
        completion(nil)
    }
}

import Foundation

protocol PriceServing {
    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void)
}

final class MockPriceService: PriceServing {
    func price(for cardID: String, completion: @escaping (PriceQuote?) -> Void) {
        completion(nil)
    }
}

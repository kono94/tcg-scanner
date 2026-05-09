import CoreGraphics
import Foundation

protocol CardRecognizing {
    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void)
}

final class StubCardRecognizer: CardRecognizing {
    func recognize(crop: CGImage, completion: @escaping (RecognitionResult?) -> Void) {
        completion(nil)
    }
}

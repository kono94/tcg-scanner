import SwiftUI

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

struct SessionView: View {
    @ObservedObject var viewModel: ScannerViewModel
    @ObservedObject var settings: ScannerSettings
    @State private var sortMode: SessionSortMode = .cardID

    private var displayedCards: [SessionCard] {
        guard !settings.allowDuplicateCards else {
            return viewModel.sessionCards
        }

        var cardsByID: [String: SessionCard] = [:]
        for card in viewModel.sessionCards {
            guard let existing = cardsByID[card.cardID] else {
                cardsByID[card.cardID] = card
                continue
            }

            if card.matchingScore > existing.matchingScore {
                cardsByID[card.cardID] = card
            }
        }
        return Array(cardsByID.values)
    }

    private var sortedCards: [SessionCard] {
        switch sortMode {
        case .cardID:
            return displayedCards.sorted {
                if $0.cardID == $1.cardID {
                    return $0.recognizedAt < $1.recognizedAt
                }
                return $0.cardID.localizedStandardCompare($1.cardID) == .orderedAscending
            }
        case .price:
            return displayedCards.sorted {
                let lhsPrice = $0.price?.sortValueUSD ?? -1
                let rhsPrice = $1.price?.sortValueUSD ?? -1
                if lhsPrice == rhsPrice {
                    return $0.cardID.localizedStandardCompare($1.cardID) == .orderedAscending
                }
                return lhsPrice > rhsPrice
            }
        }
    }

    private var totalPriceText: String {
        let totalUSD = displayedCards.reduce(0) { partialResult, card in
            partialResult + (card.price?.amountUSD ?? 0)
        }
        return PriceFormatter.displayPrice(usdAmount: totalUSD, currency: settings.priceCurrency)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Total")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(totalPriceText)
                            .font(.title2.monospacedDigit().weight(.semibold))
                    }

                    Spacer()

                    Text("\(displayedCards.count) cards")
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.top)
                .padding(.bottom, 8)

                Picker("Sort", selection: $sortMode) {
                    ForEach(SessionSortMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .padding([.horizontal, .top])

                if sortedCards.isEmpty {
                    ContentUnavailableView(
                        "No recognized cards",
                        systemImage: "rectangle.stack",
                        description: Text("Cards appear here after their track is lost or scanning stops.")
                    )
                } else {
                    List {
                        ForEach(sortedCards) { card in
                            SessionCardRow(card: card, currency: settings.priceCurrency)
                        }
                        .onDelete(perform: deleteCards)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Session")
            .toolbar {
                Button("Reset Session", role: .destructive) {
                    viewModel.resetSession()
                }
                .disabled(displayedCards.isEmpty)
            }
        }
    }

    private func deleteCards(at offsets: IndexSet) {
        if settings.allowDuplicateCards {
            let ids = Set(offsets.map { sortedCards[$0].id })
            viewModel.deleteSessionCards(ids: ids)
        } else {
            let cardIDs = Set(offsets.map { sortedCards[$0].cardID })
            viewModel.deleteSessionCards(cardIDs: cardIDs)
        }
    }
}

private struct SessionCardRow: View {
    let card: SessionCard
    let currency: PriceCurrency

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(card.cardID)
                    .font(.headline.monospacedDigit())
                Text(card.name)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                Text("Match \(Int(card.matchingScore * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            Text(card.price?.displayPrice(currency: currency) ?? "-")
                .font(.headline.monospacedDigit())
                .multilineTextAlignment(.trailing)
        }
        .padding(.vertical, 6)
    }
}

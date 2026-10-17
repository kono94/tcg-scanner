// Local card metadata and artwork-family relationships from the bundle's catalog adapter.

export interface CatalogCard {
  id: string; // namespaced identity game:language:card_id
  game: string;
  language: string;
  cardId: string;
  name: string;
  family: string; // artwork family established by the catalog adapter, not by name or base code alone
  finish: string;
  printing: string;
  /** Reference image path under the curated dataset (served locally in dev). */
  ref?: string;
  /**
   * 'back': a card-back gallery entry (id `back:<game>`), not a card. It is never matched, priced,
   * counted or offered as a candidate. Absent means a real card.
   */
  kind?: 'card' | 'back';
}

export function isBack(card: CatalogCard | undefined): boolean {
  return card?.kind === 'back';
}

export class Catalog {
  private readonly byId = new Map<string, CatalogCard>();
  private readonly byFamily = new Map<string, CatalogCard[]>();

  constructor(cards: readonly CatalogCard[]) {
    for (const card of cards) {
      this.byId.set(card.id, card);
      const members = this.byFamily.get(card.family);
      if (members) members.push(card);
      else this.byFamily.set(card.family, [card]);
    }
  }

  get size(): number {
    return this.byId.size;
  }

  get(id: string): CatalogCard | undefined {
    return this.byId.get(id);
  }

  family(id: string): CatalogCard[] {
    const card = this.byId.get(id);
    return card ? [...(this.byFamily.get(card.family) ?? [card])] : [];
  }
  /**
   * Manual lookup by name or card number (e.g. "nami", "OP01-016", "sv03 224"): one entry per artwork,
   * in the preferred language when available. Exact number matches first, then name matches.
   */
  search(query: string, preferredLanguage: string, limit = 24): CatalogCard[] {
    const norm = (t: string) => t.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
    const q = norm(query);
    if (q.length < 2) return [];
    const compact = q.replace(/ /g, '');
    const scored: [number, CatalogCard][] = [];
    for (const [, members] of this.byFamily) {
      if (members.some(isBack)) continue;
      const card = members.find((c) => c.language === preferredLanguage) ?? members[0];
      let best = 0;
      for (const c of members) {
        const id = norm(c.cardId).replace(/ /g, '');
        const name = norm(c.name);
        const s = id === compact ? 4 : id.startsWith(compact) ? 3 : name === q ? 2.5 : name.startsWith(q) ? 2 : name.includes(q) ? 1 : id.includes(compact) ? 0.5 : 0;
        best = Math.max(best, s);
      }
      if (best) scored.push([best, card]);
    }
    return scored.sort((a, b) => b[0] - a[0] || a[1].cardId.localeCompare(b[1].cardId)).slice(0, limit).map(([, c]) => c);
  }
}

export const LANGUAGE_NAMES: Record<string, string> = { en: 'English', de: 'German', ja: 'Japanese' };

export function languageName(code: string): string {
  return LANGUAGE_NAMES[code] ?? code.toUpperCase();
}

export function gameName(game: string): string {
  return game === 'onepiece' ? 'One Piece' : game === 'pokemon' ? 'Pokémon' : game;
}

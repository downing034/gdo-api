# frozen_string_literal: true

class GameOddsSerializer < BaseSerializer
  def as_json
    {
      spread: {
        favorite: object.spread_favorite_team&.code,
        value: object.spread_value&.to_f,
        favorite_odds: object.spread_favorite_odds,
        underdog_odds: object.spread_underdog_odds
      },
      total: {
        line: object.total_line&.to_f,
        over_odds: object.over_odds,
        under_odds: object.under_odds
      },
      moneyline: {
        favorite: object.moneyline_favorite_team&.code,
        favorite_odds: object.moneyline_favorite_odds,
        underdog_odds: object.moneyline_underdog_odds
      },
      fetched_at: object.fetched_at&.iso8601,
      is_opening: object.is_opening
    }
  end
end
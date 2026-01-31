# frozen_string_literal: true

FactoryBot.define do
  factory :game_odds, class: 'GameOdds' do
    game
    data_source
    spread_favorite_team { game.home_team }
    spread_value { -5.5 }
    spread_favorite_odds { -110 }
    spread_underdog_odds { -110 }
    total_line { 145.5 }
    over_odds { -110 }
    under_odds { -110 }
    moneyline_favorite_team { game.home_team }
    moneyline_favorite_odds { -150 }
    moneyline_underdog_odds { 130 }
    fetched_at { Time.current }
    is_opening { false }
  end
end
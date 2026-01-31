# frozen_string_literal: true

FactoryBot.define do
  factory :season do
    league
    sequence(:name) { |n| "#{2024 + n}-#{(25 + n).to_s.rjust(2, '0')}" }
    start_date { Date.new(2024, 11, 1) }
    end_date { Date.new(2025, 4, 15) }
    active { true }

    trait :inactive do
      active { false }
    end

    trait :nfl_2025 do
      name { '2025-26' }
      start_date { Date.new(2025, 3, 12) }
      end_date { Date.new(2026, 3, 11) }
      association :league, :nfl
    end

    trait :ncaaf_2025 do
      name { '2025-26' }
      start_date { Date.new(2025, 7, 1) }
      end_date { Date.new(2026, 3, 31) }
      association :league, :ncaaf
    end

    trait :ncaam_2025 do
      name { '2025-26' }
      start_date { Date.new(2025, 11, 1) }
      end_date { Date.new(2026, 4, 30) }
      association :league, :ncaam
    end

    trait :mlb_2025 do
      name { '2025' }
      start_date { Date.new(2025, 2, 1) }
      end_date { Date.new(2025, 11, 1) }
      active { false }
      association :league, :mlb
    end
  end
end
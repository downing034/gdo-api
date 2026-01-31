# frozen_string_literal: true

FactoryBot.define do
  factory :league do
    sport
    sequence(:code) { |n| "league_#{n}" }
    sequence(:name) { |n| "League #{n}" }
    display_name { name }
    has_conferences { true }
    active { true }

    trait :inactive do
      active { false }
    end

    trait :ncaam do
      code { 'ncaam' }
      name { "NCAA Men's Basketball" }
      display_name { 'NCAAM' }
      association :sport, :basketball
    end

    trait :mlb do
      code { 'mlb' }
      name { 'Major League Baseball' }
      display_name { 'MLB' }
      association :sport, :baseball
    end

    trait :nfl do
      code { 'nfl' }
      name { 'National Football League' }
      display_name { 'NFL' }
      association :sport, :football
    end

    trait :ncaaf do
      code { 'ncaaf' }
      name { 'NCAA Football' }
      display_name { 'NCAAF' }
      association :sport, :football
    end

    trait :nba do
      code { 'nba' }
      name { 'National Basketball Association' }
      display_name { 'NBA' }
      active { false }
      association :sport, :basketball
    end

    trait :nhl do
      code { 'nhl' }
      name { 'National Hockey League' }
      display_name { 'NHL' }
      active { false }
      association :sport, :hockey
    end
  end
end
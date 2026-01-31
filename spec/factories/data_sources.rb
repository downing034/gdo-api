# frozen_string_literal: true

FactoryBot.define do
  factory :data_source do
    sequence(:code) { |n| "source_#{n}" }
    sequence(:name) { |n| "Data Source #{n}" }
    base_url { "https://example.com" }

    trait :gdo do
      code { 'gdo' }
      name { 'Gameday Oracle' }
      base_url { '' }
    end

    trait :espn do
      code { 'espn' }
      name { 'ESPN Sports API' }
      base_url { 'https://site.api.espn.com/apis/site/v2' }
    end

    trait :sportsline do
      code { 'sportsline' }
      name { 'Sportsline' }
      base_url { 'https://www.sportsline.com/' }
    end

    trait :barttorvik do
      code { 'barttorvik' }
      name { 'Bart Torvik' }
      base_url { 'https://barttorvik.com/trankpre.php' }
    end

    trait :ken_pom do
      code { 'ken_pom' }
      name { 'Ken Pom' }
      base_url { 'https://kenpom.com' }
    end

    trait :mlb_api do
      code { 'mlb_api' }
      name { 'MLB StatsAPI' }
      base_url { 'https://statsapi.mlb.com' }
    end
  end
end
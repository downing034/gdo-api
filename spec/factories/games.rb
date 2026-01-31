# frozen_string_literal: true

FactoryBot.define do
  factory :game do
    league
    season { association :season, league: league }
    association :home_team, factory: :team
    association :away_team, factory: :team
    start_time { 1.day.from_now }
    status { 'scheduled' }
    sequence(:external_id) { |n| "espn_#{n}" }
    is_stale { false }

    trait :final do
      status { 'final' }
      start_time { 1.day.ago }
    end

    trait :in_progress do
      status { 'in_progress' }
      start_time { 1.hour.ago }
    end
  end
end
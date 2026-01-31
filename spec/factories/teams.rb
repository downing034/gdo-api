# frozen_string_literal: true

FactoryBot.define do
  factory :team do
    sequence(:code) { |n| "TEAM#{n}" }
    sequence(:location_name) { |n| "City #{n}" }
    sequence(:nickname) { |n| "Team #{n}" }
    active { true }

    trait :inactive do
      active { false }
    end

    trait :duke do
      code { 'DUKE' }
      location_name { 'Duke' }
      nickname { 'Blue Devils' }
    end

    trait :unc do
      code { 'UNC' }
      location_name { 'North Carolina' }
      nickname { 'Tar Heels' }
    end
  end
end